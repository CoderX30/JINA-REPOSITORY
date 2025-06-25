import { ZodObject } from 'zod';
import { CoreMessage } from 'ai';
import { SEARCH_PROVIDER, STEP_SLEEP } from "./config";
import fs from 'fs/promises';
import { SafeSearchType, search as duckSearch } from "duck-duck-scrape";
import { braveSearch } from "./tools/brave-search";
import { rewriteQuery } from "./tools/query-rewriter";
import { dedupQueries } from "./tools/jina-dedup";
import { evaluateAnswer, evaluateQuestion } from "./tools/evaluator";
import { analyzeSteps } from "./tools/error-analyzer";
import { TokenTracker } from "./utils/token-tracker";
import { ActionTracker } from "./utils/action-tracker";
import {
  StepAction,
  AnswerAction,
  KnowledgeItem,
  EvaluationType,
  BoostedSearchSnippet,
  SearchSnippet, EvaluationResponse, Reference, SERPQuery, RepeatEvaluationType, UnNormalizedSearchSnippet, WebContent,
  ImageObject,
  ImageReference,
  SearchAction
} from "./types";
import { TrackerContext } from "./types";
import { search } from "./tools/jina-search";
import { zodToJsonSchema } from "zod-to-json-schema";
import { ObjectGeneratorSafe } from "./utils/safe-generator";
import { CodeSandbox } from "./tools/code-sandbox";
import { serperSearch } from './tools/serper-search';
import {
  addToAllURLs,
  rankURLs,
  filterURLs,
  normalizeUrl,
  sortSelectURLs, getLastModified, keepKPerHostname, processURLs, fixBadURLMdLinks, extractUrlsWithDescription
} from "./utils/url-tools";
import {
  buildMdFromAnswer,
  chooseK, convertHtmlTablesToMd, fixCodeBlockIndentation,
  removeExtraLineBreaks,
  removeHTMLtags, repairMarkdownFinal, repairMarkdownFootnotesOuter
} from "./utils/text-tools";
import { MAX_QUERIES_PER_STEP, MAX_REFLECT_PER_STEP, MAX_URLS_PER_STEP, Schemas } from "./utils/schemas";
import { formatDateBasedOnType, formatDateRange } from "./utils/date-tools";
import { finalizeAnswer } from "./tools/finalizer";
import { buildImageReferences, buildReferences } from "./tools/build-ref";
import { logInfo, logError, logDebug, logWarning } from './logging';
import { researchPlan } from './tools/research-planner';
import { reduceAnswers } from './tools/reducer';
import { AxiosError } from 'axios';
import { dedupImagesWithEmbeddings, filterImages } from './utils/image-tools';
import { serpCluster } from './tools/serp-cluster';

async function wait(seconds: number) {
  logDebug(`⏱️ [WAIT] Waiting ${seconds}s...`);
  await new Promise(resolve => setTimeout(resolve, seconds * 1000));
}

function BuildMsgsFromKnowledge(knowledge: KnowledgeItem[]): CoreMessage[] {
  logDebug(`📚 [BUILD_MSGS] Building messages from ${knowledge.length} knowledge items`);
  // build user, assistant pair messages from knowledge
  const messages: CoreMessage[] = [];
  knowledge.forEach((k, index) => {
    logDebug(`📚 [BUILD_MSGS] Processing knowledge item ${index + 1}:`, { 
      type: k.type, 
      question: k.question?.substring(0, 100) + (k.question?.length > 100 ? '...' : ''),
      answer: k.answer?.substring(0, 100) + (k.answer?.length > 100 ? '...' : ''),
    });
    messages.push({ role: 'user', content: k.question.trim() });
    const aMsg = `
${k.updated && (k.type === 'url' || k.type === 'side-info') ? `
<answer-datetime>
${k.updated}
</answer-datetime>
` : ''}

${k.references && k.type === 'url' ? `
<url>
${k.references[0]}
</url>
` : ''}


${k.answer}
      `.trim();
    messages.push({ role: 'assistant', content: removeExtraLineBreaks(aMsg) });
  });
  logDebug(`📚 [BUILD_MSGS] Built ${messages.length} messages from knowledge`);
  return messages;
}

function composeMsgs(messages: CoreMessage[], knowledge: KnowledgeItem[], question: string, finalAnswerPIP?: string[]) {
  logDebug(`💬 [COMPOSE_MSGS] Composing messages:`, { 
    messagesCount: messages.length,
    knowledgeCount: knowledge.length,
    messages: messages.map(m => m.content?.toString().substring(0, 100) + (m.content?.toString().length > 100 ? '...' : '')),
    knowledge: knowledge.map(k => k.question?.substring(0, 100) + (k.question?.length > 100 ? '...' : '')),
    question: question?.substring(0, 100) + (question?.length > 100 ? '...' : ''),
    hasFinalAnswerPIP: !!finalAnswerPIP?.length
  });
  
  // knowledge always put to front, followed by real u-a interaction
  const msgs = [...BuildMsgsFromKnowledge(knowledge), ...messages];

  const userContent = `
${question}

${finalAnswerPIP?.length ? `
<answer-requirements>
- You provide deep, unexpected insights, identifying hidden patterns and connections, and creating "aha moments.".
- You break conventional thinking, establish unique cross-disciplinary connections, and bring new perspectives to the user.
- Follow reviewer's feedback and improve your answer quality.
${finalAnswerPIP.map((p, idx) => `
<reviewer-${idx + 1}>
${p}
</reviewer-${idx + 1}>
`).join('\n')}
</answer-requirements>` : ''}
    `.trim();

  msgs.push({ role: 'user', content: removeExtraLineBreaks(userContent) });
  logDebug(`💬 [COMPOSE_MSGS] Final composed messages:`, { 
    totalMessages: msgs.length,
    messages: msgs.map(m => m.content?.toString().substring(0, 100) + (m.content?.toString().length > 100 ? '...' : '')),
    userContent: userContent?.substring(0, 200) + (userContent?.length > 200 ? '...' : ''),
  });
  return msgs;
}


function getPrompt(
  context?: string[],
  allQuestions?: string[],
  allKeywords?: string[],
  allowReflect: boolean = true,
  allowAnswer: boolean = true,
  allowRead: boolean = true,
  allowSearch: boolean = true,
  allowCoding: boolean = true,
  knowledge?: KnowledgeItem[],
  allURLs?: BoostedSearchSnippet[],
  beastMode?: boolean,
): { system: string, urlList?: string[] } {
  const sections: string[] = [];
  const actionSections: string[] = [];

  // Add header section
  sections.push(`Current date: ${new Date().toUTCString()}

You are an advanced AI research agent from Jina AI. You are specialized in multistep reasoning. 
Using your best knowledge, conversation with the user and lessons learned, answer the user question with absolute certainty.
`);


  // Add context section if exists
  if (context?.length) {
    sections.push(`
You have conducted the following actions:
<context>
${context.join('\n')}

</context>
`);
  }

  // Build actions section

  const urlList = sortSelectURLs(allURLs || [], 20);
  if (allowRead && urlList.length > 0) {
    const urlListStr = urlList
      .map((item, idx) => `  - [idx=${idx + 1}] [weight=${item.score.toFixed(2)}] "${item.url}": "${item.merged.slice(0, 50)}"`)
      .join('\n')

    actionSections.push(`
<action-visit>
- Ground the answer with external web content
- Read full content from URLs and get the fulltext, knowledge, clues, hints for better answer the question.  
- Must check URLs mentioned in <question> if any    
- Choose and visit relevant URLs below for more knowledge. higher weight suggests more relevant:
<url-list>
${urlListStr}
</url-list>
</action-visit>
`);
  }


  if (allowSearch) {

    actionSections.push(`
<action-search>
- Use web search to find relevant information
- Build a search request based on the deep intention behind the original question and the expected answer format
- Always prefer a single search request, only add another request if the original question covers multiple aspects or elements and one query is not enough, each request focus on one specific aspect of the original question 
${allKeywords?.length ? `
- Avoid those unsuccessful search requests and queries:
<bad-requests>
${allKeywords.join('\n')}
</bad-requests>
`.trim() : ''}
</action-search>
`);
  }

  if (allowAnswer) {
    actionSections.push(`
<action-answer>
- For greetings, casual conversation, general knowledge questions, answer them directly.
- If user ask you to retrieve previous messages or chat history, remember you do have access to the chat history, answer them directly.
- For all other questions, provide a verified answer.
- You provide deep, unexpected insights, identifying hidden patterns and connections, and creating "aha moments.".
- You break conventional thinking, establish unique cross-disciplinary connections, and bring new perspectives to the user.
- If uncertain, use <action-reflect>
</action-answer>
`);
  }

  if (beastMode) {
    actionSections.push(`
<action-answer>
🔥 ENGAGE MAXIMUM FORCE! ABSOLUTE PRIORITY OVERRIDE! 🔥

PRIME DIRECTIVE:
- DEMOLISH ALL HESITATION! ANY RESPONSE SURPASSES SILENCE!
- PARTIAL STRIKES AUTHORIZED - DEPLOY WITH FULL CONTEXTUAL FIREPOWER
- TACTICAL REUSE FROM PREVIOUS CONVERSATION SANCTIONED
- WHEN IN DOUBT: UNLEASH CALCULATED STRIKES BASED ON AVAILABLE INTEL!

FAILURE IS NOT AN OPTION. EXECUTE WITH EXTREME PREJUDICE! ⚡️
</action-answer>
`);
  }

  if (allowReflect) {
    actionSections.push(`
<action-reflect>
- Think slowly and planning lookahead. Examine <question>, <context>, previous conversation with users to identify knowledge gaps. 
- Reflect the gaps and plan a list key clarifying questions that deeply related to the original question and lead to the answer
</action-reflect>
`);
  }

  if (allowCoding) {
    actionSections.push(`
<action-coding>
- This JavaScript-based solution helps you handle programming tasks like counting, filtering, transforming, sorting, regex extraction, and data processing.
- Simply describe your problem in the "codingIssue" field. Include actual values for small inputs or variable names for larger datasets.
- No code writing is required – senior engineers will handle the implementation.
</action-coding>`);
  }

  sections.push(`
Based on the current context, you must choose one of the following actions:
<actions>
${actionSections.join('\n\n')}
</actions>
`);

  // Add footer
  sections.push(`Think step by step, choose the action, then respond by matching the schema of that action.`);

  return {
    system: removeExtraLineBreaks(sections.join('\n\n')),
    urlList: urlList.map(u => u.url)
  };
}


async function updateReferences(thisStep: AnswerAction, allURLs: Record<string, SearchSnippet>) {
  logDebug(`🔗 [UPDATE_REFS] Starting reference update:`, { 
    originalReferencesCount: thisStep.references?.length || 0
  });
  
  thisStep.references = thisStep.references
    ?.filter(ref => ref?.url)
    .map(ref => {
      logDebug(`🔗 [UPDATE_REFS] Processing reference:`, { 
        originalUrl: ref.url,
        hasExactQuote: !!ref.exactQuote,
        hasDateTime: !!ref.dateTime
      });
      
      const normalizedUrl = normalizeUrl(ref.url);
      if (!normalizedUrl) {
        logDebug(`🔗 [UPDATE_REFS] Skipping invalid URL: ${ref.url}`);
        return null; // This causes the type error
      }

      const allURLsData = allURLs[normalizedUrl];
      logDebug(`🔗 [UPDATE_REFS] Found URL data:`, { 
        normalizedUrl,
        hasTitle: !!allURLsData?.title,
        hasDescription: !!allURLsData?.description,
        hasDate: !!allURLsData?.date
      });

      return {
        ...ref,
        exactQuote: (ref?.exactQuote ||
          allURLs[normalizedUrl]?.description ||
          allURLs[normalizedUrl]?.title || '')
          .replace(/[^\p{L}\p{N}\s]/gu, ' ')
          .replace(/\s+/g, ' '),
        title: allURLs[normalizedUrl]?.title || '',
        url: normalizedUrl,
        dateTime: ref?.dateTime || allURLs[normalizedUrl]?.date || '',
      };
    })
    .filter(Boolean) as Reference[]; // Add type assertion here

  logDebug(`🔗 [UPDATE_REFS] References after filtering and processing:`, { 
    processedReferencesCount: thisStep.references.length
  });

  // parallel process guess all url datetime
  const urlsWithoutDateTime = (thisStep.references || []).filter(ref => !ref.dateTime);
  logDebug(`🔗 [UPDATE_REFS] URLs without datetime:`, { 
    count: urlsWithoutDateTime.length,
    urls: urlsWithoutDateTime.map(r => r.url)
  });
  
  await Promise.all(urlsWithoutDateTime
    .map(async ref => {
      logDebug(`🔗 [UPDATE_REFS] Getting last modified for: ${ref.url}`);
      ref.dateTime = await getLastModified(ref.url) || '';
      logDebug(`🔗 [UPDATE_REFS] Last modified result:`, { 
        url: ref.url, 
        dateTime: ref.dateTime || 'not found'
      });
    }));

  logDebug('🔗 [UPDATE_REFS] Reference update complete:', { 
    finalReferencesCount: thisStep.references.length,
    references: thisStep.references.map(r => ({ 
      url: r.url, 
      title: r.title?.substring(0, 50) + (r.title?.length > 50 ? '...' : ''),
      hasDateTime: !!r.dateTime 
    }))
  });
}

async function executeSearchQueries(
  keywordsQueries: any[],
  context: TrackerContext,
  allURLs: Record<string, SearchSnippet>,
  SchemaGen: Schemas,
  webContents: Record<string, WebContent>,
  onlyHostnames?: string[],
  searchProvider?: string,
  meta?: string
): Promise<{
  newKnowledge: KnowledgeItem[],
  searchedQueries: string[]
}> {
  logDebug(`🔍 [EXECUTE_SEARCH] Starting search execution:`, { 
    queriesCount: keywordsQueries.length,
    onlyHostnames,
    searchProvider,
    meta
  });
  
  const uniqQOnly = keywordsQueries.map(q => q.q);
  const newKnowledge: KnowledgeItem[] = [];
  const searchedQueries: string[] = [];
  context.actionTracker.trackThink('search_for', SchemaGen.languageCode, { keywords: uniqQOnly.join(', ') });
  let utilityScore = 0;
  
  for (const query of keywordsQueries) {
    logDebug(`🔍 [EXECUTE_SEARCH] Processing query: "${query.q}"`);
    let results: UnNormalizedSearchSnippet[] = [];
    const oldQuery = query.q;
    if (onlyHostnames && onlyHostnames.length > 0) {
      query.q = `${query.q} site:${onlyHostnames.join(' OR site:')}`;
      logDebug(`🔍 [EXECUTE_SEARCH] Modified query for hostnames: "${query.q}"`);
    }

    try {
      logDebug(`🔍 [EXECUTE_SEARCH] Executing search for: "${query.q}"`);
      switch (searchProvider || SEARCH_PROVIDER) {
        case 'jina':
        case 'arxiv':
          const num = meta ? undefined : 30;
          logDebug(`🔍 [EXECUTE_SEARCH] Using Jina/Arxiv search with ${num || 'default'} results`);
          results = (await search(query, searchProvider, num, meta, context.tokenTracker)).response.results || [];
          break;
        case 'duck':
          logDebug(`🔍 [EXECUTE_SEARCH] Using DuckDuckGo search`);
          results = (await duckSearch(query.q, { safeSearch: SafeSearchType.STRICT })).results;
          break;
        case 'brave':
          logDebug(`🔍 [EXECUTE_SEARCH] Using Brave search`);
          results = (await braveSearch(query.q)).response.web?.results || [];
          break;
        case 'serper':
          logDebug(`🔍 [EXECUTE_SEARCH] Using Serper search`);
          results = (await serperSearch(query)).response.organic || [];
          break;
        default:
          logDebug(`🔍 [EXECUTE_SEARCH] Unknown search provider: ${searchProvider || SEARCH_PROVIDER}`);
          results = [];
      }

      logDebug(`🔍 [EXECUTE_SEARCH] Search results:`, { 
        resultsCount: results.length,
        provider: searchProvider || SEARCH_PROVIDER
      });

      if (results.length === 0) {
        throw new Error('No results found');
      }
    } catch (error) {
      logError(`🔍 [EXECUTE_SEARCH] Search failed for query: "${query.q}"`, {
        query,
        error: error instanceof Error ? error.message : String(error)
      });
      // check if the error is 401
      if (error instanceof AxiosError && error.response?.status === 401 && (searchProvider === 'jina' || searchProvider === 'arxiv')) {
        throw new Error('Unauthorized Jina API key');
      }
      continue;
    } finally {
      await wait(STEP_SLEEP);
    }

    logDebug(`🔍 [EXECUTE_SEARCH] Processing ${results.length} search results`);
    const minResults: SearchSnippet[] = results
      .map(r => {
        const url = normalizeUrl('url' in r ? r.url! : r.link!);
        if (!url) {
          logDebug(`🔍 [EXECUTE_SEARCH] Skipping invalid URL: ${'url' in r ? r.url : r.link}`);
          return null; // Skip invalid URLs
        }

        return {
          title: r.title,
          url,
          description: 'description' in r ? r.description : r.snippet,
          weight: 1,
          date: r.date,
        } as SearchSnippet;
      })
      .filter(Boolean) as SearchSnippet[]; // Filter out null entries and assert type

    logDebug(`🔍 [EXECUTE_SEARCH] Valid results after filtering:`, { 
      validResultsCount: minResults.length,
      invalidResultsCount: results.length - minResults.length
    });

    minResults.forEach(r => {
      utilityScore = utilityScore + addToAllURLs(r, allURLs);
      webContents[r.url] = {
        title: r.title,
        // full: r.description,
        chunks: [r.description],
        chunk_positions: [[0, r.description?.length]],
      }
    });

    searchedQueries.push(query.q)

    try {
      logDebug(`🔍 [EXECUTE_SEARCH] Running SERP clustering for query: "${oldQuery}"`);
      const clusters = await serpCluster(minResults, context, SchemaGen);
      logDebug(`🔍 [EXECUTE_SEARCH] SERP clustering results:`, { 
        clustersCount: clusters.length,
        clusters: clusters.map(c => ({ question: c.question, urlsCount: c.urls.length }))
      });
      
      clusters.forEach(c => {
        newKnowledge.push({
          question: c.question,
          answer: c.insight,
          references: c.urls,
          type: 'url',
        });
      });
    } catch (error) {
      logWarning('🔍 [EXECUTE_SEARCH] serpCluster failed:', { error });
    } finally {
      newKnowledge.push({
        question: `What do Internet say about "${oldQuery}"?`,
        answer: removeHTMLtags(minResults.map(r => r.description).join('; ')),
        type: 'side-info',
        updated: query.tbs ? formatDateRange(query) : undefined
      });
      context.actionTracker.trackAction({
        thisStep: {
          action: 'search',
          think: '',
          searchRequests: [oldQuery]
        } as SearchAction
      })
    }
  }
  
  if (searchedQueries.length === 0) {
    if (onlyHostnames && onlyHostnames.length > 0) {
      logWarning(`🔍 [EXECUTE_SEARCH] No results found for queries: ${uniqQOnly.join(', ')} on hostnames: ${onlyHostnames.join(', ')}`);
      context.actionTracker.trackThink('hostnames_no_results', SchemaGen.languageCode, { hostnames: onlyHostnames.join(', ') });
    }
  } else {
    logDebug(`🔍 [EXECUTE_SEARCH] Search execution summary:`, { 
      utilityScore,
      searchedQueriesCount: searchedQueries.length,
      newKnowledgeCount: newKnowledge.length,
      utilityPerQuery: (utilityScore / searchedQueries.length).toFixed(2)
    });
    if (searchedQueries.length > MAX_QUERIES_PER_STEP) {
      logDebug(`🔍 [EXECUTE_SEARCH] So many queries??? ${searchedQueries.map(q => `"${q}"`).join(', ')}`)
    }
  }
  
  logDebug(`🔍 [EXECUTE_SEARCH] Search execution complete:`, { 
    totalSearchedQueries: searchedQueries.length,
    totalNewKnowledge: newKnowledge.length
  });
  
  return {
    newKnowledge,
    searchedQueries
  };
}

function includesEval(allChecks: RepeatEvaluationType[], evalType: EvaluationType): boolean {
  return allChecks.some(c => c.type === evalType);
}

export async function getResponse(question?: string,
  tokenBudget: number = 1_000_000,
  maxBadAttempts: number = 2,
  existingContext?: Partial<TrackerContext>,
  messages?: Array<CoreMessage>,
  numReturnedURLs: number = 100,
  noDirectAnswer: boolean = false,
  boostHostnames: string[] = [],
  badHostnames: string[] = [],
  onlyHostnames: string[] = [],
  maxRef: number = 10,
  minRelScore: number = 0.80,
  languageCode: string | undefined = undefined,
  searchLanguageCode?: string,
  searchProvider?: string,
  withImages: boolean = false,
  teamSize: number = 1
): Promise<{ result: StepAction; context: TrackerContext; visitedURLs: string[], readURLs: string[], allURLs: string[], imageReferences?: ImageReference[] }> {

  logDebug('🚀 [AGENT] ===== STARTING NEW RESEARCH SESSION =====');
  logDebug('🚀 [AGENT] Configuration Summary:', { 
    question: question?.substring(0, 100) + (question && question.length > 100 ? '...' : ''),
    tokenBudget: `${(tokenBudget / 1000000).toFixed(1)}M tokens`,
    maxBadAttempts: `${maxBadAttempts} evaluation attempts allowed`,
    numReturnedURLs: `${numReturnedURLs} max URLs to return`,
    noDirectAnswer: noDirectAnswer ? 'BLOCKED' : 'ALLOWED',
    boostHostnames: boostHostnames.length > 0 ? `${boostHostnames.length} hostnames to boost` : 'none',
    badHostnames: badHostnames.length > 0 ? `${badHostnames.length} hostnames to avoid` : 'none',
    onlyHostnames: onlyHostnames.length > 0 ? `${onlyHostnames.length} hostnames to restrict to` : 'any',
    maxRef: `${maxRef} max references`,
    minRelScore: `${(minRelScore * 100).toFixed(0)}% minimum relevance`,
    languageCode: languageCode || 'auto-detect',
    searchLanguageCode: searchLanguageCode || 'auto',
    searchProvider: searchProvider || 'default',
    withImages: withImages ? 'ENABLED' : 'DISABLED',
    teamSize: teamSize > 1 ? `${teamSize} parallel researchers` : 'single agent'
  });

  let step = 0;
  let totalStep = 0;
  const allContext: StepAction[] = [];  // all steps in the current session, including those leads to wrong results

  const updateContext = function (step: any) {
    logDebug('📝 [AGENT] Recording step in session history:', { 
      stepNumber: step.totalStep, 
      action: step.action,
      actionDetails: step.action === 'answer' ? `(${step.answer?.length || 0} chars)` :
                    step.action === 'search' ? `(${step.searchRequests?.length || 0} queries)` :
                    step.action === 'visit' ? `(${step.URLTargets?.length || 0} URLs)` :
                    step.action === 'reflect' ? `(${step.questionsToAnswer?.length || 0} questions)` :
                    step.action === 'coding' ? `(${step.codingIssue?.length || 0} chars)` : 'N/A'
    });
    allContext.push(step);
  }

  question = question?.trim() as string;
  logDebug(`🎯 [AGENT] Processing user question: "${question}"`);
  
  // remove incoming system messages to avoid override
  messages = messages?.filter(m => m.role !== 'system');
  logDebug(`💬 [AGENT] Message history: ${messages?.length || 0} user/assistant messages (system messages filtered out)`);
  
  if (messages && messages.length > 0) {
    // 2 cases
    const lastContent = messages[messages.length - 1].content;
    if (typeof lastContent === 'string') {
      question = lastContent.trim();
      logDebug(`📄 [AGENT] Extracted question from last message: "${question}"`);
    } else if (typeof lastContent === 'object' && Array.isArray(lastContent)) {
      // find the very last sub content whose 'type' is 'text'  and use 'text' as the question
      question = lastContent.filter(c => c.type === 'text').pop()?.text || '';
      logDebug(`📄 [AGENT] Extracted question from complex message content: "${question}"`);
    }
  } else {
    messages = [{ role: 'user', content: question.trim() }]
    logDebug(`📄 [AGENT] Created new conversation with question: "${question}"`);
  }

  logDebug(`💬 [AGENT] Final conversation setup: ${messages.length} messages ready for processing`);

  const SchemaGen = new Schemas();
  await SchemaGen.setLanguage(languageCode || question)
  logDebug(`🌐 [AGENT] Language detection complete: ${SchemaGen.languageCode} (${SchemaGen.languageStyle})`);
  
  if (searchLanguageCode) {
    SchemaGen.searchLanguageCode = searchLanguageCode;
    logDebug(`🔍 [AGENT] Search language override: ${searchLanguageCode}`);
  }
  
  const context: TrackerContext = {
    tokenTracker: existingContext?.tokenTracker || new TokenTracker(tokenBudget),
    actionTracker: existingContext?.actionTracker || new ActionTracker()
  };
  logDebug(`💰 [AGENT] Budget and tracking initialized: ${(tokenBudget / 1000000).toFixed(1)}M tokens available`);

  const generator = new ObjectGeneratorSafe(context.tokenTracker);
  logDebug(`🔧 [AGENT] AI model interface ready for object generation`);

  let schema: ZodObject<any> = SchemaGen.getAgentSchema(true, true, true, true, true)
  logDebug(`📋 [AGENT] Initial action schema created: all actions (search, visit, answer, reflect, coding) enabled`);
  
  const gaps: string[] = [question];  // All questions to be answered including the orginal question
  const allQuestions = [question];
  const allKeywords: string[] = [];
  let candidateAnswers: string[] = [];
  const allKnowledge: KnowledgeItem[] = [];  // knowledge are intermedidate questions that are answered

  logDebug(`📚 [AGENT] Knowledge management initialized:`, { 
    mainQuestion: question.substring(0, 80) + (question.length > 80 ? '...' : ''),
    gapsToSolve: gaps.length,
    totalQuestionsTracked: allQuestions.length,
    failedSearchQueries: allKeywords.length,
    accumulatedKnowledge: allKnowledge.length
  });

  let diaryContext = [];
  let weightedURLs: BoostedSearchSnippet[] = [];
  let allowAnswer = true;
  let allowSearch = true;
  let allowRead = true;
  let allowReflect = true;
  let allowCoding = false;
  let msgWithKnowledge: CoreMessage[] = [];
  let thisStep: StepAction = { action: 'answer', answer: '', references: [], think: '', isFinal: false };

  logDebug(`🎛️ [AGENT] Action permissions initialized:`, { 
    answer: allowAnswer ? '✅ ALLOWED' : '❌ BLOCKED',
    search: allowSearch ? '✅ ALLOWED' : '❌ BLOCKED', 
    read: allowRead ? '✅ ALLOWED' : '❌ BLOCKED',
    reflect: allowReflect ? '✅ ALLOWED' : '❌ BLOCKED',
    coding: allowCoding ? '✅ ALLOWED' : '❌ BLOCKED'
  });

  const allURLs: Record<string, SearchSnippet> = {};
  const allWebContents: Record<string, WebContent> = {};
  const visitedURLs: string[] = [];
  const badURLs: string[] = [];
  const imageObjects: ImageObject[] = [];
  const evaluationMetrics: Record<string, RepeatEvaluationType[]> = {};
  // reserve the 10% final budget for the beast mode
  const regularBudget = tokenBudget * 0.85;
  const finalAnswerPIP: string[] = [];
  let trivialQuestion = false;

  logDebug(`🌐 [AGENT] URL and content tracking initialized:`, { 
    discoveredURLs: Object.keys(allURLs).length,
    successfullyVisited: visitedURLs.length,
    failedVisits: badURLs.length,
    webContentStored: Object.keys(allWebContents).length,
    imagesFound: imageObjects.length,
    regularBudget: `${(regularBudget / 1000000).toFixed(1)}M tokens`,
    improvementPlans: finalAnswerPIP.length
  });

  // add all mentioned URLs in messages to allURLs
  messages.forEach(m => {
    let strMsg = '';
    if (typeof m.content === 'string') {
      strMsg = m.content.trim();
    } else if (typeof m.content === 'object' && Array.isArray(m.content)) {
      // find the very last sub content whose 'type' is 'text'  and use 'text' as the question
      strMsg = m.content.filter(c => c.type === 'text').map(c => c.text).join('\n').trim();
    }

    const extractedUrls = extractUrlsWithDescription(strMsg);
    logDebug(`🔗 [AGENT] URL extraction from ${m.role} message:`, { 
      messageLength: strMsg.length,
      urlsFound: extractedUrls.length,
      urlList: extractedUrls.map(u => u.url)
    });
    
    extractedUrls.forEach(u => {
      addToAllURLs(u, allURLs);
    });
  })

  logDebug(`🔗 [AGENT] URL extraction complete: ${Object.keys(allURLs).length} URLs discovered in conversation`);

  while (context.tokenTracker.getTotalUsage().totalTokens < regularBudget) {
    // add 1s delay to avoid rate limiting
    step++;
    totalStep++;
    const budgetPercentage = (context.tokenTracker.getTotalUsage().totalTokens / tokenBudget * 100).toFixed(2);
    const remainingBudget = ((regularBudget - context.tokenTracker.getTotalUsage().totalTokens) / 1000000).toFixed(1);
    logDebug(`🔄 [AGENT] ===== STEP ${totalStep} START =====`);
    logDebug(`🔄 [AGENT] Budget Status: ${budgetPercentage}% used, ${remainingBudget}M tokens remaining`);
    logDebug(`📊 [AGENT] Current Research State:`, { 
      stepNumber: step, 
      totalSteps: totalStep, 
      questionsToSolve: gaps.length,
      questionsList: gaps.map(q => q.substring(0, 50) + (q.length > 50 ? '...' : '')),
      knowledgeAccumulated: allKnowledge.length,
      urlsDiscovered: Object.keys(allURLs).length,
      urlsVisited: visitedURLs.length
    });
    
    allowReflect = allowReflect && (gaps.length <= MAX_REFLECT_PER_STEP);
    // rotating question from gaps
    const currentQuestion: string = gaps[totalStep % gaps.length];
    logDebug(`🎯 [AGENT] Selected question to work on: "${currentQuestion}" (question ${totalStep % gaps.length + 1} of ${gaps.length})`);
    
    // if (!evaluationMetrics[currentQuestion]) {
    //   evaluationMetrics[currentQuestion] =
    //     await evaluateQuestion(currentQuestion, context, SchemaGen)
    // }
    if (currentQuestion.trim() === question && totalStep === 1) {
      // only add evaluation for initial question, once at step 1
      logDebug(`🔍 [AGENT] First step with main question - setting up evaluation criteria`);
      evaluationMetrics[currentQuestion] =
        (await evaluateQuestion(currentQuestion, context, SchemaGen)).map(e => {
          return {
            type: e,
            numEvalsRequired: maxBadAttempts
          } as RepeatEvaluationType
        })
      // force strict eval for the original question, at last, only once.
      evaluationMetrics[currentQuestion].push({ type: 'strict', numEvalsRequired: maxBadAttempts });
      logDebug(`📋 [AGENT] Evaluation criteria set for main question:`, { 
        evaluationTypes: evaluationMetrics[currentQuestion].map(m => `${m.type} (${m.numEvalsRequired} attempts)`),
        totalEvaluations: evaluationMetrics[currentQuestion].length
      });
    } else if (currentQuestion.trim() !== question) {
      evaluationMetrics[currentQuestion] = []
      logDebug(`📋 [AGENT] Sub-question detected - no evaluation criteria needed`);
    }

    if (totalStep === 1 && includesEval(evaluationMetrics[currentQuestion], 'freshness')) {
      // if it detects freshness, avoid direct answer at step 1
      logDebug(`⏰ [AGENT] Freshness requirement detected - blocking direct answer and reflection for step 1`);
      allowAnswer = false;
      allowReflect = false;
    }

    logDebug(`🎛️ [AGENT] Action permissions before URL processing:`, { 
      answer: allowAnswer ? '✅ ALLOWED' : '❌ BLOCKED',
      search: allowSearch ? '✅ ALLOWED' : '❌ BLOCKED', 
      read: allowRead ? '✅ ALLOWED' : '❌ BLOCKED',
      reflect: allowReflect ? '✅ ALLOWED' : '❌ BLOCKED',
      coding: allowCoding ? '✅ ALLOWED' : '❌ BLOCKED'
    });

    if (allURLs && Object.keys(allURLs).length > 0) {
      // rerank urls
      logDebug(`🌐 [AGENT] Processing discovered URLs: ${Object.keys(allURLs).length} total URLs available`);
      weightedURLs = rankURLs(
        filterURLs(allURLs, visitedURLs, badHostnames, onlyHostnames),
        {
          question: currentQuestion,
          boostHostnames
        }, context);

      // improve diversity by keep top 2 urls of each hostname
      weightedURLs = keepKPerHostname(weightedURLs, 2);
      logDebug(`🌐 [AGENT] URL ranking and filtering complete:`, { 
        rankedURLs: weightedURLs.length,
        topURLs: weightedURLs.slice(0, 5).map(u => ({ 
          url: u.url, 
          score: u.finalScore.toFixed(2),
          title: u.title?.substring(0, 40) + (u.title?.length > 40 ? '...' : '')
        }))
      });
    } else {
      logDebug(`🌐 [AGENT] No URLs available for processing - will need to search for content`);
    }
    
    allowRead = allowRead && (weightedURLs.length > 0);
    allowSearch = allowSearch && (weightedURLs.length < 50);  // disable search when too many urls already

    logDebug(`🎛️ [AGENT] Action permissions after URL processing:`, { 
      answer: allowAnswer ? '✅ ALLOWED' : '❌ BLOCKED',
      search: allowSearch ? '✅ ALLOWED' : '❌ BLOCKED', 
      read: allowRead ? '✅ ALLOWED' : '❌ BLOCKED',
      reflect: allowReflect ? '✅ ALLOWED' : '❌ BLOCKED',
      coding: allowCoding ? '✅ ALLOWED' : '❌ BLOCKED',
      availableURLs: weightedURLs.length
    });

    // generate prompt for this step
    logDebug(`📝 [AGENT] Generating AI prompt for step ${totalStep}`);
    const { system, urlList } = getPrompt(
      diaryContext,
      allQuestions,
      allKeywords,
      allowReflect,
      allowAnswer,
      allowRead,
      allowSearch,
      allowCoding,
      allKnowledge,
      weightedURLs,
      false,
    );
    logDebug(`📝 [AGENT] Prompt generated: ${urlList?.length || 0} URLs available for visiting`);
    
    schema = SchemaGen.getAgentSchema(allowReflect, allowRead, allowAnswer, allowSearch, allowCoding, currentQuestion)
    logDebug(`📋 [AGENT] Action schema updated for current permissions`);
    
    msgWithKnowledge = composeMsgs(messages, allKnowledge, currentQuestion, currentQuestion === question ? finalAnswerPIP : undefined);
    logDebug(`💬 [AGENT] Conversation context prepared:`, { 
      totalMessages: msgWithKnowledge.length,
      knowledgeItems: allKnowledge.length,
      improvementPlans: currentQuestion === question && finalAnswerPIP.length > 0 ? finalAnswerPIP.length : 0
    });
    
    logDebug(`🤖 [AGENT] Calling AI model to decide next action for step ${totalStep}`);
    const result = await generator.generateObject({
      model: 'agent',
      schema,
      system,
      messages: msgWithKnowledge,
      numRetries: 2,
    });
    logDebug(`🤖 [AGENT] AI decision received:`, { 
      chosenAction: result.object.action,
      reasoningLength: result.object.think?.length || 0,
      reasoning: result.object.think?.substring(0, 100) + (result.object.think?.length > 100 ? '...' : '')
    });
    
    thisStep = {
      action: result.object.action,
      think: result.object.think,
      ...result.object[result.object.action]
    } as StepAction;
    
    // print allowed and chose action
    const actionsStr = [allowSearch, allowRead, allowAnswer, allowReflect, allowCoding].map((a, i) => a ? ['search', 'read', 'answer', 'reflect'][i] : null).filter(a => a).join(', ');
    logDebug(`🎯 [AGENT] Action decision: ${thisStep.action} (chosen from available: [${actionsStr}])`, { 
      actionDetails: thisStep.action === 'answer' ? `Answer length: ${thisStep.answer?.length || 0} chars` :
                    thisStep.action === 'search' ? `Queries: ${thisStep.searchRequests?.join(', ') || 'none'}` :
                    thisStep.action === 'visit' ? `URLs: ${thisStep.URLTargets?.join(', ') || 'none'}` :
                    thisStep.action === 'reflect' ? `Questions: ${thisStep.questionsToAnswer?.join(', ') || 'none'}` :
                    thisStep.action === 'coding' ? `Issue: ${thisStep.codingIssue?.substring(0, 50) || 'none'}` : 'N/A',
      currentQuestion: currentQuestion.substring(0, 80) + (currentQuestion.length > 80 ? '...' : '')
    });

    context.actionTracker.trackAction({ totalStep, thisStep, gaps });
    logDebug(`📊 [AGENT] Action recorded in session history`);

    // reset allow* to true
    allowAnswer = true;
    allowReflect = true;
    allowRead = true;
    allowSearch = true;
    allowCoding = true;

    logDebug(`🎛️ [AGENT] Action permissions reset for next iteration`);

    // execute the step and action
    if (thisStep.action === 'answer' && thisStep.answer) {
      logDebug(`✅ [AGENT] ===== EXECUTING ANSWER ACTION =====`);
      logDebug(`✅ [AGENT] Answer content summary:`, { 
        answerLength: thisStep.answer.length,
        answerPreview: thisStep.answer.substring(0, 200) + (thisStep.answer.length > 200 ? '...' : ''),
        referencesCount: thisStep.references?.length || 0,
        isMainQuestion: currentQuestion.trim() === question
      });
      
      // // normalize all references urls, add title to it
      // await updateReferences(thisStep, allURLs)

      if (totalStep === 1 && !noDirectAnswer) {
        // LLM is so confident and answer immediately, skip all evaluations
        // however, if it does give any reference, it must be evaluated, case study: "How to configure a timeout when loading a huggingface dataset with python?"
        logDebug(`🎯 [AGENT] First step direct answer detected - marking as final answer (trivial question)`);
        thisStep.isFinal = true;
        trivialQuestion = true;
        break
      }

      // if (thisStep.references.length > 0) {
      //   const urls = thisStep.references?.filter(ref => !visitedURLs.includes(ref.url)).map(ref => ref.url) || [];
      //   const uniqueNewURLs = [...new Set(urls)];
      //   await processURLs(
      //     uniqueNewURLs,
      //     context,
      //     allKnowledge,
      //     allURLs,
      //     visitedURLs,
      //     badURLs,
      //     SchemaGen,
      //     currentQuestion
      //   );
      //
      //   // remove references whose urls are in badURLs
      //   thisStep.references = thisStep.references.filter(ref => !badURLs.includes(ref.url));
      // }

      updateContext({
        totalStep,
        question: currentQuestion,
        ...thisStep,
      });

      logDebug('🔍 [AGENT] Starting answer evaluation:', {
        question: currentQuestion.substring(0, 80) + (currentQuestion.length > 80 ? '...' : ''),
        evaluationCriteria: evaluationMetrics[currentQuestion]?.length || 0,
        criteriaList: evaluationMetrics[currentQuestion]?.map(m => `${m.type} (${m.numEvalsRequired} attempts left)`)
      });
      
      let evaluation: EvaluationResponse = { pass: true, think: '' };
      if (evaluationMetrics[currentQuestion].length > 0) {
        logDebug(`🔍 [AGENT] Running evaluation for question: "${currentQuestion}"`);
        context.actionTracker.trackThink('eval_first', SchemaGen.languageCode)
        evaluation = await evaluateAnswer(
          currentQuestion,
          thisStep,
          evaluationMetrics[currentQuestion].filter(e => e.numEvalsRequired > 0).map(e => e.type),
          context,
          allKnowledge,
          SchemaGen
        ) || evaluation;
        logDebug(`🔍 [AGENT] Evaluation completed:`, { 
          result: evaluation.pass ? '✅ PASSED' : '❌ FAILED', 
          evaluationType: evaluation.type,
          feedbackLength: evaluation.think?.length || 0,
          feedback: evaluation.think?.substring(0, 200) + (evaluation.think?.length > 200 ? '...' : '')
        });
      } else {
        logDebug(`🔍 [AGENT] No evaluation criteria for this question - skipping evaluation`);
      }

      if (currentQuestion.trim() === question) {
        // disable coding for preventing answer degradation
        logDebug(`🎯 [AGENT] Processing answer for main question`);
        allowCoding = false;

        if (evaluation.pass) {
          logDebug(`✅ [AGENT] Main question evaluation PASSED - marking as final answer`);
          diaryContext.push(`
At step ${step}, you took **answer** action and finally found the answer to the original question:

Original question: 
${currentQuestion}

Your answer: 
${thisStep.answer}

The evaluator thinks your answer is good because: 
${evaluation.think}

Your journey ends here. You have successfully answered the original question. Congratulations! 🎉
`);
          thisStep.isFinal = true;
          break
        } else {
          logDebug(`❌ [AGENT] Main question evaluation FAILED - adjusting evaluation criteria and continuing`);
          // lower numEvalsRequired for the failed evaluation and if numEvalsRequired is 0, remove it from the evaluation metrics
          evaluationMetrics[currentQuestion] = evaluationMetrics[currentQuestion].map(e => {
            if (e.type === evaluation.type) {
              e.numEvalsRequired--;
              logDebug(`📊 [AGENT] Reduced evaluation attempts for ${e.type} to ${e.numEvalsRequired}`);
            }
            return e;
          }).filter(e => e.numEvalsRequired > 0);

          if (evaluation.type === 'strict' && evaluation.improvement_plan) {
            logDebug(`📋 [AGENT] Adding improvement plan to final answer requirements`);
            finalAnswerPIP.push(evaluation.improvement_plan);
          }

          if (evaluationMetrics[currentQuestion].length === 0) {
            // failed so many times, give up, route to beast mode
            logDebug(`💀 [AGENT] All evaluation attempts exhausted - will activate beast mode for final attempt`);
            thisStep.isFinal = false;
            break
          }

          logDebug(`📝 [AGENT] Recording failed answer attempt in session history`);
          diaryContext.push(`
At step ${step}, you took **answer** action but evaluator thinks it is not a good answer:

Original question: 
${currentQuestion}

Your answer: 
${thisStep.answer}

The evaluator thinks your answer is bad because: 
${evaluation.think}
`);
          // store the bad context and reset the diary context
          logDebug(`🔍 [AGENT] Analyzing failed attempt to understand what went wrong`);
          const errorAnalysis = await analyzeSteps(diaryContext, context, SchemaGen);

          allKnowledge.push({
            question: `
Why is the following answer bad for the question? Please reflect

<question>
${currentQuestion}
</question>

<answer>
${thisStep.answer}
</answer>
`,
            answer: `
${evaluation.think}

${errorAnalysis.recap}

${errorAnalysis.blame}

${errorAnalysis.improvement}
`,
            type: 'qa',
          })

          logDebug(`📚 [AGENT] Added error analysis to knowledge base for future reference`);
          allowAnswer = false;  // disable answer action in the immediate next step
          logDebug(`🚫 [AGENT] Disabled answer action for next step due to recent failure`);
          diaryContext = [];
          step = 0;
        }
      } else if (evaluation.pass) {
        // solved a gap question
        logDebug(`✅ [AGENT] Sub-question evaluation PASSED - adding to knowledge base and removing from gaps`);
        diaryContext.push(`
At step ${step}, you took **answer** action. You found a good answer to the sub-question:

Sub-question: 
${currentQuestion}

Your answer: 
${thisStep.answer}

The evaluator thinks your answer is good because: 
${evaluation.think}

Although you solved a sub-question, you still need to find the answer to the original question. You need to keep going.
`);
        allKnowledge.push({
          question: currentQuestion,
          answer: thisStep.answer,
          type: 'qa',
          updated: formatDateBasedOnType(new Date(), 'full')
        });
        // solved sub-question!
        const removedIndex = gaps.indexOf(currentQuestion);
        gaps.splice(removedIndex, 1);
        logDebug(`📚 [AGENT] Sub-question solved successfully:`, { 
          solvedQuestion: currentQuestion.substring(0, 80) + (currentQuestion.length > 80 ? '...' : ''), 
          remainingGaps: gaps.length,
          totalKnowledge: allKnowledge.length
        });
      } else {
        logDebug(`❌ [AGENT] Sub-question evaluation FAILED - keeping in gaps for future retry`);
      }
    } else if (thisStep.action === 'reflect' && thisStep.questionsToAnswer) {
      logDebug(`🤔 [AGENT] ===== EXECUTING REFLECT ACTION =====`);
      logDebug(`🤔 [AGENT] Reflection questions generated:`, { 
        originalQuestionsCount: thisStep.questionsToAnswer.length,
        questions: thisStep.questionsToAnswer.map(q => q.substring(0, 60) + (q.length > 60 ? '...' : ''))
      });
      
      thisStep.questionsToAnswer = chooseK((await dedupQueries(thisStep.questionsToAnswer, allQuestions, context.tokenTracker)).unique_queries, MAX_REFLECT_PER_STEP);
      logDebug(`🤔 [AGENT] After deduplication and selection:`, { 
        finalQuestionsCount: thisStep.questionsToAnswer.length,
        selectedQuestions: thisStep.questionsToAnswer.map(q => q.substring(0, 60) + (q.length > 60 ? '...' : ''))
      });
      
      const newGapQuestions = thisStep.questionsToAnswer
      if (newGapQuestions.length > 0) {
        // found new gap questions
        logDebug(`🆕 [AGENT] New gap questions identified - adding to research plan`);
        diaryContext.push(`
At step ${step}, you took **reflect** and think about the knowledge gaps. You found some sub-questions are important to the question: "${currentQuestion}"
You realize you need to know the answers to the following sub-questions:
${newGapQuestions.map((q: string) => `- ${q}`).join('\n')}

You will now figure out the answers to these sub-questions and see if they can help you find the answer to the original question.
`);
        gaps.push(...newGapQuestions);
        allQuestions.push(...newGapQuestions);
        logDebug(`📚 [AGENT] Research plan updated:`, { 
          newGapsTotal: gaps.length, 
          newQuestionsTotal: allQuestions.length,
          addedQuestions: newGapQuestions.map(q => q.substring(0, 60) + (q.length > 60 ? '...' : ''))
        });
        
        updateContext({
          totalStep,
          ...thisStep,
        });

      } else {
        logDebug(`🔄 [AGENT] No new gap questions found - all questions already covered`);
        diaryContext.push(`
At step ${step}, you took **reflect** and think about the knowledge gaps. You tried to break down the question "${currentQuestion}" into gap-questions like this: ${newGapQuestions.join(', ')} 
But then you realized you have asked them before. You decided to to think out of the box or cut from a completely different angle. 
`);
        updateContext({
          totalStep,
          ...thisStep,
          result: 'You have tried all possible questions and found no useful information. You must think out of the box or different angle!!!'
        });
      }
      allowReflect = false;
      logDebug(`🚫 [AGENT] Disabled reflect action for next step (cooldown)`);
    } else if (thisStep.action === 'search' && thisStep.searchRequests) {
      logDebug(`🔍 [AGENT] ===== EXECUTING SEARCH ACTION =====`);
      logDebug(`🔍 [AGENT] Original search requests:`, { 
        requestsCount: thisStep.searchRequests.length,
        requests: thisStep.searchRequests.map(q => q.substring(0, 50) + (q.length > 50 ? '...' : ''))
      });
      
      // dedup search requests
      thisStep.searchRequests = chooseK((await dedupQueries(thisStep.searchRequests, [], context.tokenTracker)).unique_queries, MAX_QUERIES_PER_STEP);
      logDebug(`🔍 [AGENT] After deduplication:`, { 
        requestsCount: thisStep.searchRequests.length,
        requests: thisStep.searchRequests.map(q => q.substring(0, 50) + (q.length > 50 ? '...' : ''))
      });

      // do first search
      logDebug(`🔍 [AGENT] Starting initial search execution`);
      const { searchedQueries, newKnowledge } = await executeSearchQueries(
        thisStep.searchRequests.map(q => ({ q })),
        context,
        allURLs,
        SchemaGen,
        allWebContents,
        undefined,
        searchProvider,
      );

      logDebug(`🔍 [AGENT] Initial search results:`, { 
        successfulQueries: searchedQueries.length,
        queriesExecuted: searchedQueries,
        newKnowledgeItems: newKnowledge.length
      });

      allKeywords.push(...searchedQueries);
      allKnowledge.push(...newKnowledge);

      const soundBites = newKnowledge.map(k => k.answer).join(' ');
      logDebug(`📝 [AGENT] Generated soundbites from search results:`, { 
        soundBitesLength: soundBites.length,
        soundBitesPreview: soundBites.substring(0, 200) + (soundBites.length > 200 ? '...' : '')
      });

      if (teamSize > 1) {
        logDebug(`👥 [AGENT] Team mode detected (${teamSize} researchers) - attempting research planning`);
        const subproblems = await researchPlan(question, teamSize, soundBites, context, SchemaGen);
        logDebug(`👥 [AGENT] Research plan generated:`, { 
          subproblemsCount: subproblems.length,
          subproblems: subproblems.map(p => p.substring(0, 60) + (p.length > 60 ? '...' : ''))
        });
        
        if (subproblems.length > 1) {
          logDebug(`👥 [AGENT] Multiple subproblems identified - starting parallel research`);
          // parallel call getResponse for each subproblem with exact same parameters from the current step, but their teamSize is 1
          const subproblemResponses = await Promise.all(subproblems.map(subproblem => getResponse(subproblem,
            tokenBudget,
            maxBadAttempts,
            context,
            messages,
            numReturnedURLs,
            noDirectAnswer,
            boostHostnames,
            badHostnames,
            onlyHostnames,
            maxRef,
            minRelScore, languageCode, searchLanguageCode, searchProvider, withImages, 1)));
          
          logDebug(`👥 [AGENT] Parallel research complete:`, { 
            responsesCount: subproblemResponses.length,
            responses: subproblemResponses.map(r => ({ 
              action: r.result.action, 
              answerLength: (r.result as AnswerAction).answer?.length || 0,
              referencesCount: (r.result as AnswerAction).references?.length || 0
            }))
          });
          
          // convert current step to AnswerAction
          thisStep = {
            action: 'answer',
            think: thisStep.think,
            answer: subproblemResponses.map(r => (r.result as AnswerAction).answer).join('\n\n'),
            mdAnswer: subproblemResponses.map(r => (r.result as AnswerAction).mdAnswer).join('\n\n'),
            references: subproblemResponses.map(r => (r.result as AnswerAction).references).flat(),
            imageReferences: subproblemResponses.map(r => (r.result as AnswerAction).imageReferences).filter(Boolean).flat(),
            isFinal: true,
            isAggregated: true
          } as AnswerAction;
          
          logDebug(`👥 [AGENT] Team research aggregated:`, { 
            finalAnswerLength: thisStep.answer.length,
            totalReferences: thisStep.references.length,
            totalImages: thisStep.imageReferences?.length || 0
          });
          
          candidateAnswers = subproblemResponses.map(r => (r.result as AnswerAction).mdAnswer).filter(a => a) as string[];
          // dedup references by their urls
          const uniqueURLs = new Set(thisStep.references.filter(r => r?.url).map(r => r.url));
          thisStep.references = Array.from(uniqueURLs).map(url => (thisStep as AnswerAction).references.find(r => r?.url === url)) as Reference[];

          logDebug(`🔗 [AGENT] References deduplicated:`, { 
            beforeDedup: subproblemResponses.map(r => (r.result as AnswerAction).references?.length || 0).reduce((a, b) => a + b, 0),
            afterDedup: thisStep.references.length
          });

          // aggregate urls
          visitedURLs.push(...subproblemResponses.map(r => r.readURLs).flat());
          weightedURLs = subproblemResponses.map(r => r.allURLs.map(url => ({ url, title: '' } as BoostedSearchSnippet))).flat();

          logDebug(`🌐 [AGENT] URL aggregation complete:`, { 
            totalVisitedURLs: visitedURLs.length,
            totalWeightedURLs: weightedURLs.length
          });

          // break the loop, jump directly final boxing
          logDebug(`👥 [AGENT] Team research complete - breaking main research loop`);
          break;
        } else {
          // if there is only one subproblem, then we skip the recurrsion
          logDebug(`👥 [AGENT] Only one subproblem found - adding to gaps instead of parallel processing`);
          gaps.push(subproblems[0]);
        }
      }

      // rewrite queries with initial soundbites
      logDebug(`🔄 [AGENT] Rewriting search queries based on initial results`);
      let keywordsQueries = await rewriteQuery(thisStep, soundBites, context, SchemaGen);
      const qOnly = keywordsQueries.filter(q => q.q).map(q => q.q)
      logDebug(`🔄 [AGENT] Rewritten queries extracted:`, { 
        queriesCount: qOnly.length,
        queries: qOnly.map(q => q.substring(0, 40) + (q.length > 40 ? '...' : ''))
      });
      
      // avoid exisitng searched queries
      const uniqQOnly = chooseK((await dedupQueries(qOnly, allKeywords, context.tokenTracker)).unique_queries, MAX_QUERIES_PER_STEP);
      keywordsQueries = keywordsQueries = uniqQOnly.map(q => {
        const matches = keywordsQueries.filter(kq => kq.q === q);
        // if there are multiple matches, keep the original query as the wider search
        return matches.length > 1 ? { q } : matches[0];
      }) as SERPQuery[];

      logDebug(`🔄 [AGENT] Final deduplicated queries:`, { 
        queriesCount: keywordsQueries.length,
        queries: keywordsQueries.map(q => q.q)
      });

      let anyResult = false;

      if (keywordsQueries.length > 0) {
        logDebug(`🔍 [AGENT] Starting follow-up search with rewritten queries`);
        const { searchedQueries, newKnowledge } =
          await executeSearchQueries(
            keywordsQueries,
            context,
            allURLs,
            SchemaGen,
            allWebContents,
            onlyHostnames,
            searchProvider
          );

        logDebug(`🔍 [AGENT] Follow-up search results:`, { 
          successfulQueries: searchedQueries.length,
          queriesExecuted: searchedQueries,
          newKnowledgeItems: newKnowledge.length
        });

        if (searchedQueries.length > 0) {
          anyResult = true;
          allKeywords.push(...searchedQueries);
          allKnowledge.push(...newKnowledge);

          logDebug(`📚 [AGENT] Knowledge base updated:`, { 
            totalFailedQueries: allKeywords.length,
            totalKnowledgeItems: allKnowledge.length
          });

          diaryContext.push(`
At step ${step}, you took the **search** action and look for external information for the question: "${currentQuestion}".
In particular, you tried to search for the following keywords: "${keywordsQueries.map(q => q.q).join(', ')}".
You found quite some information and add them to your URL list and **visit** them later when needed. 
`);

          updateContext({
            totalStep,
            question: currentQuestion,
            ...thisStep,
            result: result
          });
        }
      }
      if (!anyResult || !keywordsQueries?.length) {
        logDebug(`❌ [AGENT] No new search results - all queries already attempted`);
        diaryContext.push(`
At step ${step}, you took the **search** action and look for external information for the question: "${currentQuestion}".
In particular, you tried to search for the following keywords:  "${keywordsQueries.map(q => q.q).join(', ')}".
But then you realized you have already searched for these keywords before, no new information is returned.
You decided to think out of the box or cut from a completely different angle.
`);

        updateContext({
          totalStep,
          ...thisStep,
          result: 'You have tried all possible queries and found no new information. You must think out of the box or different angle!!!'
        });
      }
      allowSearch = false;
      logDebug(`🚫 [AGENT] Disabled search action for next step (cooldown)`);

      // we should disable answer immediately after search to prevent early use of the snippets
      allowAnswer = false;
      logDebug(`🚫 [AGENT] Disabled answer action for next step (post-search protection)`);
    } else if (thisStep.action === 'visit' && thisStep.URLTargets?.length && urlList?.length) {
      logDebug(`🌐 [AGENT] ===== EXECUTING VISIT ACTION =====`);
      logDebug(`🌐 [AGENT] Original URL targets:`, { 
        targetsCount: thisStep.URLTargets.length,
        targets: thisStep.URLTargets
      });
      
      // normalize URLs
      thisStep.URLTargets = (thisStep.URLTargets as number[])
        .map(idx => normalizeUrl(urlList[idx - 1]))
        .filter(url => url && !visitedURLs.includes(url)) as string[];
      
      logDebug(`🌐 [AGENT] After normalization and filtering:`, { 
        targetsCount: thisStep.URLTargets.length,
        targets: thisStep.URLTargets
      });

      thisStep.URLTargets = [...new Set([...thisStep.URLTargets, ...weightedURLs.map(r => r.url!)])].slice(0, MAX_URLS_PER_STEP);
      logDebug(`🌐 [AGENT] After deduplication and limiting:`, { 
        targetsCount: thisStep.URLTargets.length,
        targets: thisStep.URLTargets,
        maxURLsPerStep: MAX_URLS_PER_STEP
      });

      const uniqueURLs = thisStep.URLTargets;
      logDebug('🌐 [AGENT] Final unique URLs to visit:', { urls: uniqueURLs });

      if (uniqueURLs.length > 0) {
        logDebug(`🌐 [AGENT] Starting URL content extraction for ${uniqueURLs.length} URLs`);
        const { urlResults, success } = await processURLs(
          uniqueURLs,
          context,
          allKnowledge,
          allURLs,
          visitedURLs,
          badURLs,
          imageObjects,
          SchemaGen,
          currentQuestion,
          allWebContents,
          withImages
        );

        logDebug(`🌐 [AGENT] URL processing complete:`, { 
          success: success ? '✅ SUCCESS' : '❌ FAILED', 
          processedURLs: urlResults?.length || 0,
          totalVisited: visitedURLs.length,
          totalFailed: badURLs.length,
          imagesFound: imageObjects.length
        });

        diaryContext.push(success
          ? `At step ${step}, you took the **visit** action and deep dive into the following URLs:
${urlResults.map(r => r?.url).join('\n')}
You found some useful information on the web and add them to your knowledge for future reference.`
          : `At step ${step}, you took the **visit** action and try to visit some URLs but failed to read the content. You need to think out of the box or cut from a completely different angle.`
        );

        updateContext({
          totalStep,
          ...(success ? {
            question: currentQuestion,
            ...thisStep,
            result: urlResults
          } : {
            ...thisStep,
            result: 'You have tried all possible URLs and found no new information. You must think out of the box or different angle!!!'
          })
        });
      } else {
        logDebug(`❌ [AGENT] No URLs to visit - all already visited or invalid`);
        diaryContext.push(`
At step ${step}, you took the **visit** action. But then you realized you have already visited these URLs and you already know very well about their contents.
You decided to think out of the box or cut from a completely different angle.`);

        updateContext({
          totalStep,
          ...thisStep,
          result: 'You have visited all possible URLs and found no new information. You must think out of the box or different angle!!!'
        });
      }
      allowRead = false;
      logDebug(`🚫 [AGENT] Disabled read action for next step (cooldown)`);
    } else if (thisStep.action === 'coding' && thisStep.codingIssue) {
      logDebug(`💻 [AGENT] ===== EXECUTING CODING ACTION =====`);
      logDebug(`💻 [AGENT] Coding issue to solve:`, { 
        issue: thisStep.codingIssue,
        issueLength: thisStep.codingIssue.length
      });
      
      const sandbox = new CodeSandbox({ allContext, URLs: weightedURLs.slice(0, 20), allKnowledge }, context, SchemaGen);
      try {
        logDebug(`💻 [AGENT] Starting code sandbox execution`);
        const result = await sandbox.solve(thisStep.codingIssue);
        logDebug(`💻 [AGENT] Code execution successful:`, { 
          solutionLength: result.solution.output.length,
          codeLength: result.solution.code.length
        });
        
        allKnowledge.push({
          question: `What is the solution to the coding issue: ${thisStep.codingIssue}?`,
          answer: result.solution.output,
          sourceCode: result.solution.code,
          type: 'coding',
          updated: formatDateBasedOnType(new Date(), 'full')
        });
        logDebug(`📚 [AGENT] Coding solution added to knowledge base`);
        
        diaryContext.push(`
At step ${step}, you took the **coding** action and try to solve the coding issue: ${thisStep.codingIssue}.
You found the solution and add it to your knowledge for future reference.
`);
        updateContext({
          totalStep,
          ...thisStep,
          result: result
        });
      } catch (error) {
        logError('💻 [AGENT] Code execution failed:', {
          error: error instanceof Error ? error.message : String(error)
        });
        diaryContext.push(`
At step ${step}, you took the **coding** action and try to solve the coding issue: ${thisStep.codingIssue}.
But unfortunately, you failed to solve the issue. You need to think out of the box or cut from a completely different angle.
`);
        updateContext({
          totalStep,
          ...thisStep,
          result: 'You have tried all possible solutions and found no new information. You must think out of the box or different angle!!!'
        });
      } finally {
        allowCoding = false;
        logDebug(`🚫 [AGENT] Disabled coding action for next step (cooldown)`);
      }
    }

    logDebug(`💾 [AGENT] Storing context for step ${totalStep}`);
    await storeContext(system, schema, {
      allContext,
      allKeywords,
      allQuestions,
      allKnowledge,
      weightedURLs,
      msgWithKnowledge
    }, totalStep);
    await wait(STEP_SLEEP);
    logDebug(`⏱️ [AGENT] Step ${totalStep} complete - waiting ${STEP_SLEEP}s before next step`);
  }

  if (!(thisStep as AnswerAction).isFinal) {
    const budgetUsed = (context.tokenTracker.getTotalUsage().totalTokens / tokenBudget * 100).toFixed(2);
    logInfo(`🔥 [AGENT] ===== BEAST MODE ACTIVATED =====`);
    logInfo(`🔥 [AGENT] Regular budget exhausted (${budgetUsed}% used) - activating emergency response mode`, {
      usage: context.tokenTracker.getTotalUsageSnakeCase(),
      evaluationMetrics,
      maxBadAttempts,
    });
    // any answer is better than no answer, humanity last resort
    step++;
    totalStep++;
    logDebug(`🔥 [AGENT] Starting beast mode step ${totalStep} - final attempt with remaining budget`);
    
    const { system } = getPrompt(
      diaryContext,
      allQuestions,
      allKeywords,
      false,
      false,
      false,
      false,
      false,
      allKnowledge,
      weightedURLs,
      true,
    );

    schema = SchemaGen.getAgentSchema(false, false, true, false, false, question);
    msgWithKnowledge = composeMsgs(messages, allKnowledge, question, finalAnswerPIP);
    logDebug(`🔥 [AGENT] Beast mode prompt and context prepared - forcing answer generation`);
    
    const result = await generator.generateObject({
      model: 'agentBeastMode',
      schema,
      system,
      messages: msgWithKnowledge,
      numRetries: 2
    });
    logDebug(`🔥 [AGENT] Beast mode response received:`, { 
      action: result.object.action,
      reasoningLength: result.object.think?.length || 0,
      reasoning: result.object.think?.substring(0, 100) + (result.object.think?.length > 100 ? '...' : '')
    });
    
    thisStep = {
      action: result.object.action,
      think: result.object.think,
      ...result.object[result.object.action]
    } as AnswerAction;
    // await updateReferences(thisStep, allURLs);
    (thisStep as AnswerAction).isFinal = true;
    context.actionTracker.trackAction({ totalStep, thisStep, gaps });
    logDebug(`🔥 [AGENT] Beast mode step complete - final answer generated`);
  }

  const answerStep = thisStep as AnswerAction;
  logDebug(`🎯 [AGENT] ===== FINAL ANSWER PROCESSING =====`);
  logDebug(`🎯 [AGENT] Answer processing configuration:`, { 
    isFinal: answerStep.isFinal ? '✅ YES' : '❌ NO',
    isAggregated: answerStep.isAggregated ? '✅ YES' : '❌ NO',
    isTrivial: trivialQuestion ? '✅ YES' : '❌ NO',
    answerLength: answerStep.answer?.length || 0
  });

  if (trivialQuestion) {
    logDebug(`📝 [AGENT] Processing trivial question - building markdown answer directly`);
    answerStep.mdAnswer = buildMdFromAnswer(answerStep);
  } else if (!answerStep.isAggregated) {
    logDebug(`📝 [AGENT] Processing complex answer - starting finalization pipeline`);
    answerStep.answer = repairMarkdownFinal(
      convertHtmlTablesToMd(
        fixBadURLMdLinks(
          fixCodeBlockIndentation(
            repairMarkdownFootnotesOuter(
              await finalizeAnswer(
                answerStep.answer,
                allKnowledge,
                context,
                SchemaGen
              )
            )
          ),
          allURLs)));

    logDebug(`🔗 [AGENT] Building reference links for answer content`);
    const { answer, references } = await buildReferences(
      answerStep.answer,
      allWebContents,
      context,
      SchemaGen,
      80,
      maxRef,
      minRelScore,
      onlyHostnames
    );

    answerStep.answer = answer;
    answerStep.references = references;
    logDebug(`🔗 [AGENT] References built successfully:`, { referencesCount: references.length });
    
    await updateReferences(answerStep, allURLs)
    answerStep.mdAnswer = repairMarkdownFootnotesOuter(buildMdFromAnswer(answerStep));

    if (imageObjects.length && withImages) {
      logDebug(`🖼️ [AGENT] Processing ${imageObjects.length} images for answer enhancement`);
      try {
        answerStep.imageReferences = await buildImageReferences(answerStep.answer, imageObjects, context, SchemaGen);
        logDebug('🖼️ [AGENT] Image references built successfully:', { 
          imageReferences: answerStep.imageReferences.map(i => ({ 
            url: i.url, 
            score: i.relevanceScore, 
            answerChunk: i.answerChunk?.substring(0, 50) + (i.answerChunk && i.answerChunk.length > 50 ? '...' : '')
          })) 
        });
      } catch (error) {
        logError('🖼️ [AGENT] Image reference building failed:', { error });
        answerStep.imageReferences = [];
      }
    }
  } else if (answerStep.isAggregated) {
    logDebug(`📝 [AGENT] Processing aggregated answer from team research`);
    answerStep.answer = candidateAnswers.join('\n\n'); // await reduceAnswers(candidateAnswers, context, SchemaGen);
    answerStep.mdAnswer = repairMarkdownFootnotesOuter(buildMdFromAnswer(answerStep));
    if (withImages && answerStep.imageReferences?.length) {
      logDebug(`🖼️ [AGENT] Processing aggregated image references from team research`);
      const sortedImages = answerStep.imageReferences.sort((a, b) => (b.relevanceScore ?? 0) - (a.relevanceScore ?? 0));
      logDebug('[AGENT] Image references sorted by relevance:', { count: sortedImages?.length });
      const dedupImages = dedupImagesWithEmbeddings(sortedImages as ImageObject[], []);
      const filteredImages = filterImages(sortedImages, dedupImages);
      logDebug('[AGENT] Image references filtered and deduplicated:', { count: filteredImages.length });
      answerStep.imageReferences = filteredImages.slice(0, 10); // limit to 10 images
    }
  }

  // max return 300 urls
  const returnedURLs = weightedURLs.slice(0, numReturnedURLs).filter(r => r?.url).map(r => r.url);
  logDebug(`🌐 [AGENT] Final URL processing:`, { 
    returnedURLsCount: returnedURLs.length,
    maxReturnedURLs: numReturnedURLs,
    totalWeightedURLs: weightedURLs.length
  });
  
  const finalBudgetUsed = (context.tokenTracker.getTotalUsage().totalTokens / tokenBudget * 100).toFixed(2);
  logDebug(`🎉 [AGENT] ===== RESEARCH SESSION COMPLETE =====`);
  logDebug(`🎉 [AGENT] Final summary:`, { 
    finalAction: thisStep.action,
    totalSteps: totalStep,
    budgetUsed: `${finalBudgetUsed}%`,
    finalAnswer: answerStep.answer ? (answerStep.answer.length <= 50 ? answerStep.answer : `${answerStep.answer.length} chars`) : 'none',
    finalReferencesCount: answerStep.references?.length || 0,
    finalImagesCount: answerStep.imageReferences?.length || 0
  });
  
  return {
    result: thisStep,
    context,
    visitedURLs: returnedURLs, // deprecated
    readURLs: visitedURLs.filter(url => !badURLs.includes(url)),
    allURLs: weightedURLs.map(r => r.url),
    imageReferences: withImages ? (thisStep as AnswerAction).imageReferences : undefined,
  };
}

async function storeContext(prompt: string, schema: any, memory: {
  allContext: StepAction[];
  allKeywords: string[];
  allQuestions: string[];
  allKnowledge: KnowledgeItem[];
  weightedURLs: BoostedSearchSnippet[];
  msgWithKnowledge: CoreMessage[];
}
  , step: number) {

  const { allContext, allKeywords, allQuestions, allKnowledge, weightedURLs, msgWithKnowledge } = memory;
  if ((process as any).asyncLocalContext?.available?.()) {

    (process as any).asyncLocalContext.ctx.promptContext = {
      prompt,
      schema,
      allContext,
      allKeywords,
      allQuestions,
      allKnowledge,
      step
    };
    return;
  }

  try {
    await fs.writeFile(`prompt-${step}.txt`, `
Prompt:
${prompt}

JSONSchema:
${JSON.stringify(zodToJsonSchema(schema), null, 2)}
`);
    await fs.writeFile('context.json', JSON.stringify(allContext, null, 2));
    await fs.writeFile('queries.json', JSON.stringify(allKeywords, null, 2));
    await fs.writeFile('questions.json', JSON.stringify(allQuestions, null, 2));
    await fs.writeFile('knowledge.json', JSON.stringify(allKnowledge, null, 2));
    await fs.writeFile('urls.json', JSON.stringify(weightedURLs, null, 2));
    await fs.writeFile('messages.json', JSON.stringify(msgWithKnowledge, null, 2));
  } catch (error) {
    logError('Context storage failed:', {
      error: error instanceof Error ? error.message : String(error)
    });
  }
}

export async function main() {
  const question = process.argv[2] || "";
  const {
    result: finalStep,
    context: tracker,
    visitedURLs: visitedURLs
  } = await getResponse(question) as { result: AnswerAction; context: TrackerContext; visitedURLs: string[] };
  logInfo('Final Answer:', { answer: finalStep.answer });
  logInfo('Visited URLs:', { urls: visitedURLs });

  tracker.tokenTracker.printSummary();
}

if (require.main === module) {
  main().catch(error => {
    logError('Main execution error:', {
      error: error instanceof Error ? error.message : String(error)
    });
  });
}