import { EventEmitter } from 'events';

import { TokenUsage } from '../types';
import { LanguageModelUsage } from "ai";
import { logInfo, logError, logDebug, logWarning } from '../logging';

const TOOL_NAME = 'token-tracker';

export class TokenTracker extends EventEmitter {
  private usages: TokenUsage[] = [];
  private budget?: number;

  constructor(budget?: number) {
    super();
    /*logDebug(`🏦 [${TOOL_NAME}] ===== INITIALIZING TOKEN TRACKER =====`);
    logDebug(`🏦 [${TOOL_NAME}] TokenTracker initialization:`, { 
      budgetProvided: !!budget,
      budgetAmount: budget ? `${(budget / 1000000).toFixed(1)}M tokens` : 'unlimited',
      budgetRaw: budget
    });*/
    
    this.budget = budget;

    if ('asyncLocalContext' in process) {
      /*logDebug(`🔄 [${TOOL_NAME}] AsyncLocalContext detected, setting up charge tracking`); */
      const asyncLocalContext = process.asyncLocalContext as any;
      this.on('usage', () => {
        if (asyncLocalContext.available()) {
          const totalTokens = this.getTotalUsage().totalTokens;
          asyncLocalContext.ctx.chargeAmount = totalTokens;
          /*logDebug(`🔄 [${TOOL_NAME}] AsyncLocalContext charge updated:`, { 
            chargeAmount: totalTokens,
            formattedCharge: `${(totalTokens / 1000).toFixed(1)}K tokens`
          });*/
        }
      });
    } else {
      /*logDebug(`🔄 [${TOOL_NAME}] AsyncLocalContext not available, skipping charge tracking setup`); */
    }

    /*logDebug(`✅ [${TOOL_NAME}] TokenTracker initialized successfully`); */
  }

  trackUsage(tool: string, usage: LanguageModelUsage) {
    /*logDebug(`📊 [${TOOL_NAME}] ===== TRACKING TOKEN USAGE =====`);*/
    /*logDebug(`📊 [${TOOL_NAME}] New usage tracking:`, { 
      tool: tool,
      promptTokens: usage.promptTokens,
      completionTokens: usage.completionTokens,
      totalTokens: usage.totalTokens,
      costRatio: usage.completionTokens > 0 ? (usage.completionTokens / usage.totalTokens * 100).toFixed(1) + '% completion' : '0% completion'
    });*/

    const u = { tool, usage };
    this.usages.push(u);
    
    const currentTotal = this.getTotalUsage();
    /*logDebug(`💰 [${TOOL_NAME}] Usage tracking complete:`, { 
      tool: tool,
      sessionUsages: this.usages.length,
      totalSessionTokens: currentTotal.totalTokens,
      budgetUsed: this.budget ? `${(currentTotal.totalTokens / this.budget * 100).toFixed(2)}%` : 'N/A',
      remainingBudget: this.budget ? `${((this.budget - currentTotal.totalTokens) / 1000000).toFixed(1)}M tokens` : 'unlimited'
    });*/

    if (this.budget && currentTotal.totalTokens > this.budget * 0.8) {
      /*logWarning(`⚠️ [${TOOL_NAME}] Token budget approaching limit:`, { 
        currentUsage: currentTotal.totalTokens,
        budget: this.budget,
        percentageUsed: (currentTotal.totalTokens / this.budget * 100).toFixed(2) + '%',
        tokensRemaining: this.budget - currentTotal.totalTokens
      });*/
    }

    if (this.budget && currentTotal.totalTokens > this.budget) {
      /*logWarning(`🚨 [${TOOL_NAME}] Token budget exceeded:`, { 
        currentUsage: currentTotal.totalTokens,
        budget: this.budget,
        overage: currentTotal.totalTokens - this.budget,
        percentageUsed: (currentTotal.totalTokens / this.budget * 100).toFixed(2) + '%'
      });*/
    }

    this.emit('usage', usage);
    /*logDebug(`📡 [${TOOL_NAME}] Usage event emitted for tool: ${tool}`);*/
  }

  getTotalUsage(): LanguageModelUsage {
    /*logDebug(`📈 [${TOOL_NAME}] Calculating total usage across all tracked tools`);*/ 
    
    const totalUsage = this.usages.reduce((acc, { usage }) => {
      acc.promptTokens += usage.promptTokens;
      acc.completionTokens += usage.completionTokens;
      acc.totalTokens += usage.totalTokens;
      return acc;
    }, { promptTokens: 0, completionTokens: 0, totalTokens: 0 });

    /*logDebug(`📈 [${TOOL_NAME}] Total usage calculated:`, { 
      promptTokens: totalUsage.promptTokens,
      completionTokens: totalUsage.completionTokens,
      totalTokens: totalUsage.totalTokens,
      sessionsTracked: this.usages.length,
      averageTokensPerSession: this.usages.length > 0 ? (totalUsage.totalTokens / this.usages.length).toFixed(1) : '0'
    });*/

    return totalUsage;
  }

  getTotalUsageSnakeCase(): { prompt_tokens: number, completion_tokens: number, total_tokens: number } {
    /*logDebug(`📈 [${TOOL_NAME}] Calculating total usage in snake_case format for API compatibility`);*/
    
    const snakeCaseUsage = this.usages.reduce((acc, { usage }) => {
      acc.prompt_tokens += usage.promptTokens;
      acc.completion_tokens += usage.completionTokens;
      acc.total_tokens += usage.totalTokens;
      return acc;
    }, { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 });

    /*logDebug(`📈 [${TOOL_NAME}] Snake case usage calculated:`, { 
      prompt_tokens: snakeCaseUsage.prompt_tokens,
      completion_tokens: snakeCaseUsage.completion_tokens,
      total_tokens: snakeCaseUsage.total_tokens,
      formattedTotal: `${(snakeCaseUsage.total_tokens / 1000).toFixed(1)}K tokens`
    });*/

    return snakeCaseUsage;
  }

  getUsageBreakdown(): Record<string, number> {
    /*logDebug(`📊 [${TOOL_NAME}] Generating usage breakdown by tool`);*/
    
    const breakdown = this.usages.reduce((acc, { tool, usage }) => {
      acc[tool] = (acc[tool] || 0) + usage.totalTokens;
      return acc;
    }, {} as Record<string, number>);

    const sortedBreakdown = Object.entries(breakdown)
      .sort(([,a], [,b]) => b - a)
      .reduce((acc, [tool, tokens]) => {
        acc[tool] = tokens;
        return acc;
      }, {} as Record<string, number>);

    /*logDebug(`📊 [${TOOL_NAME}] Usage breakdown by tool (sorted by usage):`, { 
      toolCount: Object.keys(sortedBreakdown).length,
      topTools: Object.entries(sortedBreakdown).slice(0, 5).map(([tool, tokens]) => ({ 
        tool, 
        tokens, 
        percentage: this.getTotalUsage().totalTokens > 0 ? (tokens / this.getTotalUsage().totalTokens * 100).toFixed(1) + '%' : '0%'
      })),
      breakdown: sortedBreakdown
    });*/

    return sortedBreakdown;
  }

  printSummary() {
    /*logInfo(`💰 [${TOOL_NAME}] ===== TOKEN USAGE SUMMARY =====`);*/
    const breakdown = this.getUsageBreakdown();
    const totalUsage = this.getTotalUsage();
    
      /*logInfo(`💰 [${TOOL_NAME}] Session Summary:`, {
      budget: this.budget ? `${(this.budget / 1000000).toFixed(1)}M tokens` : 'unlimited',
      budgetUsed: this.budget ? `${(totalUsage.totalTokens / this.budget * 100).toFixed(2)}%` : 'N/A',
      total: totalUsage,
      totalFormatted: `${(totalUsage.totalTokens / 1000000).toFixed(3)}M tokens`,
      promptFormatted: `${(totalUsage.promptTokens / 1000000).toFixed(3)}M tokens`,
      completionFormatted: `${(totalUsage.completionTokens / 1000000).toFixed(3)}M tokens`,
      breakdown
    });*/

    // Log efficiency metrics
    if (totalUsage.totalTokens > 0) {
      const efficiency = {
        promptRatio: (totalUsage.promptTokens / totalUsage.totalTokens * 100).toFixed(1) + '% prompt',
        completionRatio: (totalUsage.completionTokens / totalUsage.totalTokens * 100).toFixed(1) + '% completion',
        averagePerSession: this.usages.length > 0 ? `${(totalUsage.totalTokens / this.usages.length / 1000).toFixed(1)}K tokens/session` : '0K tokens/session',
        totalSessions: this.usages.length
      };
      
      /*logInfo(`📊 [${TOOL_NAME}] Efficiency Metrics:`, efficiency);*/
    }

    if (this.budget) {
      const budgetAnalysis = {
        remainingTokens: this.budget - totalUsage.totalTokens,
        remainingFormatted: `${((this.budget - totalUsage.totalTokens) / 1000000).toFixed(1)}M tokens`,
        budgetUtilization: (totalUsage.totalTokens / this.budget * 100).toFixed(2) + '%',
        isOverBudget: totalUsage.totalTokens > this.budget
      };
      
      /*logInfo(`🏦 [${TOOL_NAME}] Budget Analysis:`, budgetAnalysis);*/
      
      if (budgetAnalysis.isOverBudget) {
        /*logWarning(`🚨 [${TOOL_NAME}] BUDGET EXCEEDED:`, { 
          overage: totalUsage.totalTokens - this.budget,
          overageFormatted: `${((totalUsage.totalTokens - this.budget) / 1000000).toFixed(1)}M tokens over budget`
        });*/
      }
    }

    /*logInfo(`💰 [${TOOL_NAME}] ===== END TOKEN USAGE SUMMARY =====`);*/
  }

  reset() {
    /*logDebug(`🔄 [${TOOL_NAME}] ===== RESETTING TOKEN TRACKER =====`);*/
    const previousUsageCount = this.usages.length;
    const previousTotalTokens = this.getTotalUsage().totalTokens;
    
    this.usages = [];
    
    /*logDebug(`🔄 [${TOOL_NAME}] Token tracker reset complete:`, { 
      previousUsageCount,
      previousTotalTokens,
      previousTotalFormatted: `${(previousTotalTokens / 1000000).toFixed(3)}M tokens`,
      currentUsageCount: this.usages.length,
      currentTotalTokens: this.getTotalUsage().totalTokens
    });*/
    
    /*logInfo(`🔄 [${TOOL_NAME}] Token tracker reset - cleared ${previousUsageCount} usage records totaling ${(previousTotalTokens / 1000000).toFixed(3)}M tokens`);*/
  }
}
