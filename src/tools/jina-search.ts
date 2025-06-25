import { TokenTracker } from "../utils/token-tracker";
import { JinaSearchResponse, SERPQuery } from '../types';
import { JINA_API_KEY } from "../config";
import axiosClient from '../utils/axios-client';
import { logError, logDebug } from '../logging';

export async function search(
  query: SERPQuery,
  domain?: string,
  num?: number,
  meta?: string,
  tracker?: TokenTracker
): Promise<{ response: JinaSearchResponse }> {
  try {
    if (domain !== 'arxiv') {
      domain = undefined;  // default to general search
    }

    const { data } = await axiosClient.post<JinaSearchResponse>(
      `https://svip.jina.ai/`,
      {
        ...query,
        domain,
        num,
        meta
      },
      {
        headers: {
          'Accept': 'application/json',
          'Authorization': `Bearer ${JINA_API_KEY}`,
        },
        timeout: 10000,
        responseType: 'json'
      }
    );

    if (!data.results || !Array.isArray(data.results)) {
      throw new Error('Invalid response format');
    }

    logDebug('Search results metadata:', { metadata: data.meta });

    const tokenTracker = tracker || new TokenTracker();
    tokenTracker.trackUsage('search', {
      totalTokens: data.meta.credits,
      promptTokens: query.q.length,
      completionTokens: 0
    });

    return { response: data };
  } catch (error) {
    logError('Search error:', { error });
    throw error;
  }
}