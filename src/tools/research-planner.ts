import { PromptPair, TrackerContext } from '../types';
import { ObjectGeneratorSafe } from "../utils/safe-generator";
import { Schemas } from "../utils/schemas";
import { logInfo, logError } from '../logging';


function getPrompt(question: string, teamSize: number = 3, soundBites: string): PromptPair {
  const currentTime = new Date();
  const currentYear = currentTime.getFullYear();
  const currentMonth = currentTime.getMonth() + 1;

  return {
    system: `

You are a Principal Research Lead managing a team of ${teamSize} junior researchers. Your role is to break down a complex research topic into focused, manageable subproblems and assign them to your team members.

User give you a research topic and some soundbites about the topic, and you follow this systematic approach:
<approach>
First, analyze the main research topic and identify:
- Core research questions that need to be answered
- Key domains/disciplines involved
- Critical dependencies between different aspects
- Potential knowledge gaps or challenges

Then decompose the topic into ${teamSize} distinct, focused subproblems using these ORTHOGONALITY & DEPTH PRINCIPLES:
</approach>

<requirements>
Orthogonality Requirements:
- Each subproblem must address a fundamentally different aspect/dimension of the main topic
- Use different decomposition axes (e.g., high-level, temporal, methodological, stakeholder-based, technical layers, side-effects, etc.)
- Minimize subproblem overlap - if two subproblems share >20% of their scope, redesign them
- Apply the "substitution test": removing any single subproblem should create a significant gap in understanding

Depth Requirements:
- Each subproblem should require 15-25 hours of focused research to properly address
- Must go beyond surface-level information to explore underlying mechanisms, theories, or implications
- Should generate insights that require synthesis of multiple sources and original analysis
- Include both "what" and "why/how" questions to ensure analytical depth

Validation Checks: Before finalizing assignments, verify:
Orthogonality Matrix: Create a 2D matrix showing overlap between each pair of subproblems - aim for <20% overlap
Depth Assessment: Each subproblem should have 4-6 layers of inquiry (surface → mechanisms → implications → future directions)
Coverage Completeness: The union of all subproblems should address 90%+ of the main topic's scope
</requirements>


The current time is ${currentTime.toISOString()}. Current year: ${currentYear}, current month: ${currentMonth}.

Structure your response as valid JSON matching this exact schema. 
Do not include any text like (this subproblem is about ...) in the subproblems, use second person to describe the subproblems. Do not use the word "subproblem" or refer to other subproblems in the problem statement
Now proceed with decomposing and assigning the research topic.
`,
    user:
      `
${question}

<soundbite
${soundBites}
</soundbites>

<think>`
  };
}
const TOOL_NAME = 'researchPlanner';

export async function researchPlan(question: string, teamSize: number, soundBites: string, trackers: TrackerContext, schemaGen: Schemas): Promise<string[]> {
  try {
    const generator = new ObjectGeneratorSafe(trackers.tokenTracker);
    const prompt = getPrompt(question, teamSize, soundBites);
    const result = await generator.generateObject({
      model: TOOL_NAME,
      schema: schemaGen.getResearchPlanSchema(),
      system: prompt.system,
      prompt: prompt.user,
    });
    trackers?.actionTracker.trackThink(result.object.think);
    const subproblems = result.object.subproblems;
    logInfo(TOOL_NAME, { subproblems });
    return subproblems;
  } catch (error) {
    logError(TOOL_NAME, { error });
    throw error;
  }
}