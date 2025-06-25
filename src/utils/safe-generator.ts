import { z } from 'zod';
import {
  CoreMessage,
  generateObject,
  LanguageModelUsage,
  NoObjectGeneratedError,
  Schema
} from "ai";
import { TokenTracker } from "./token-tracker";
import { getModel, ToolName, getToolConfig } from "../config";
import Hjson from 'hjson'; // Import Hjson library for more lenient JSON parsing
import { logError, logDebug, logWarning } from '../logging';

/**
 * Result interface for object generation operations
 * Contains the generated object and token usage information
 */
interface GenerateObjectResult<T> {
  object: T;
  usage: LanguageModelUsage;
}

/**
 * Options interface for object generation
 * Defines all parameters needed for the generation process
 */
interface GenerateOptions<T> {
  model: ToolName;                    // Which AI model to use
  schema: z.ZodType<T> | Schema<T>;  // Schema to validate against
  prompt?: string;                    // Optional prompt text
  system?: string;                    // Optional system message
  messages?: CoreMessage[];           // Optional conversation messages
  numRetries?: number;                // Number of retry attempts
}

/**
 * Safe Object Generator with multiple fallback mechanisms
 * Handles AI model failures gracefully with schema distillation and manual parsing
 */
export class ObjectGeneratorSafe {
  private tokenTracker: TokenTracker;

  constructor(tokenTracker?: TokenTracker) {
    this.tokenTracker = tokenTracker || new TokenTracker();
    logDebug(`🔧 [SAFE_GEN] ObjectGeneratorSafe initialized with token tracker`);
  }

  /**
   * Creates a distilled version of a schema by removing all descriptions
   * This makes the schema simpler for fallback parsing scenarios
   * Used when the main model fails to generate valid JSON
   */
  private createDistilledSchema<T>(schema: z.ZodType<T> | Schema<T>): z.ZodType<T> | Schema<T> {
    logDebug(`🔧 [SAFE_GEN] Creating distilled schema for fallback parsing`);
    
    // For zod schemas - strip descriptions to simplify parsing
    if (schema instanceof z.ZodType) {
      logDebug(`🔧 [SAFE_GEN] Processing Zod schema for distillation`);
      return this.stripZodDescriptions(schema);
    }

    // For AI SDK Schema objects - strip descriptions to simplify parsing
    if (typeof schema === 'object' && schema !== null) {
      logDebug(`🔧 [SAFE_GEN] Processing AI SDK Schema for distillation`);
      return this.stripSchemaDescriptions(schema as Schema<T>);
    }

    // If we can't determine the schema type, return as is
    logDebug(`🔧 [SAFE_GEN] Unknown schema type, returning as-is`);
    return schema;
  }

  /**
   * Recursively strips descriptions from Zod schemas
   * This simplifies the schema for better fallback parsing success
   */
  private stripZodDescriptions<T>(zodSchema: z.ZodType<T>): z.ZodType<T> {
    logDebug(`🔧 [SAFE_GEN] Stripping descriptions from Zod schema: ${zodSchema.constructor.name}`);
    
    // Handle ZodObject - strip descriptions from all properties
    if (zodSchema instanceof z.ZodObject) {
      const shape = zodSchema._def.shape();
      const newShape: Record<string, any> = {};

      for (const key in shape) {
        if (Object.prototype.hasOwnProperty.call(shape, key)) {
          logDebug(`🔧 [SAFE_GEN] Processing object property: ${key}`);
          // Recursively strip descriptions from nested schemas
          newShape[key] = this.stripZodDescriptions(shape[key]);
        }
      }

      return z.object(newShape) as unknown as z.ZodType<T>;
    }

    // Handle ZodArray - strip descriptions from array elements
    if (zodSchema instanceof z.ZodArray) {
      logDebug(`🔧 [SAFE_GEN] Processing array schema`);
      return z.array(this.stripZodDescriptions(zodSchema._def.type)) as unknown as z.ZodType<T>;
    }

    // Handle ZodString - create clean string schema without metadata
    if (zodSchema instanceof z.ZodString) {
      logDebug(`🔧 [SAFE_GEN] Processing string schema`);
      // Create a new string schema without any describe() metadata
      return z.string() as unknown as z.ZodType<T>;
    }

    // Handle complex schemas (unions, intersections) - return as-is for now
    if (zodSchema instanceof z.ZodUnion || zodSchema instanceof z.ZodIntersection) {
      logDebug(`🔧 [SAFE_GEN] Complex schema detected (union/intersection), returning as-is`);
      // These are more complex schemas that would need special handling
      // This is a simplified implementation
      return zodSchema;
    }

    // For other primitive types or complex types we're not handling specifically,
    // return as is
    logDebug(`🔧 [SAFE_GEN] Unknown Zod schema type, returning as-is`);
    return zodSchema;
  }

  /**
   * Strips descriptions from AI SDK Schema objects
   * This simplifies the schema for better fallback parsing success
   */
  private stripSchemaDescriptions<T>(schema: Schema<T>): Schema<T> {
    logDebug(`🔧 [SAFE_GEN] Stripping descriptions from AI SDK Schema`);
    
    // Deep clone the schema to avoid modifying the original
    const clonedSchema = JSON.parse(JSON.stringify(schema));

    // Recursively remove description properties
    const removeDescriptions = (obj: any) => {
      if (typeof obj !== 'object' || obj === null) return;

      // Remove descriptions from object properties
      if (obj.properties) {
        for (const key in obj.properties) {
          // Remove description property
          if (obj.properties[key].description) {
            logDebug(`🔧 [SAFE_GEN] Removing description from property: ${key}`);
            delete obj.properties[key].description;
          }

          // Recursively process nested properties
          removeDescriptions(obj.properties[key]);
        }
      }

      // Handle arrays - remove descriptions from array items
      if (obj.items) {
        if (obj.items.description) {
          logDebug(`🔧 [SAFE_GEN] Removing description from array items`);
          delete obj.items.description;
        }
        removeDescriptions(obj.items);
      }

      // Handle any other nested objects that might contain descriptions
      if (obj.anyOf) obj.anyOf.forEach(removeDescriptions);
      if (obj.allOf) obj.allOf.forEach(removeDescriptions);
      if (obj.oneOf) obj.oneOf.forEach(removeDescriptions);
    };

    removeDescriptions(clonedSchema);
    logDebug(`🔧 [SAFE_GEN] Schema descriptions stripped successfully`);
    return clonedSchema;
  }

  /**
   * Main object generation method with comprehensive error handling and fallbacks
   * Implements a multi-tier fallback strategy for robust object generation
   */
  async generateObject<T>(options: GenerateOptions<T>): Promise<GenerateObjectResult<T>> {
    const {
      model,
      schema,
      prompt,
      system,
      messages,
      numRetries = 0,
    } = options;

    logDebug(`🚀 [SAFE_GEN] ===== STARTING OBJECT GENERATION =====`);
    logDebug(`🚀 [SAFE_GEN] Generation parameters:`, { 
      model: model, 
      hasSchema: !!schema ? '✅ YES' : '❌ NO', 
      prompt: prompt ? (prompt.substring(0,200) + (prompt.length > 200 ? '...' : '')) : '❌ NO',
      system: system ? (system.substring(0,200) + (system.length > 200 ? '...' : '')) : '❌ NO',
      messagesCount: messages?.length || 0,
      retryAttempts: numRetries
    });

    if (!model || !schema) {
      logError(`🚀 [SAFE_GEN] Missing required parameters:`, { 
        model: !!model ? '✅ PROVIDED' : '❌ MISSING', 
        schema: !!schema ? '✅ PROVIDED' : '❌ MISSING' 
      });
      throw new Error('Model and schema are required parameters');
    }

    try {
      // 🎯 PRIMARY ATTEMPT: Try with main model and original schema
      logDebug(`🎯 [SAFE_GEN] Primary attempt: Using ${model} model with original schema`);
      const result = await generateObject({
        model: getModel(model),
        schema,
        prompt,
        system,
        messages,
        maxTokens: getToolConfig(model).maxTokens,
        temperature: getToolConfig(model).temperature,
      });

      logDebug(`✅ [SAFE_GEN] Primary attempt successful:`, { 
        model: model, 
        usage: result.usage,
        tokensUsed: result.usage.totalTokens
      });
      this.tokenTracker.trackUsage(model, result.usage);
      return result;

    } catch (error: unknown) {
      logWarning(`❌ [SAFE_GEN] Primary attempt failed for ${model} model:`, { 
        error: error instanceof Error ? error.message : String(error),
        errorType: error instanceof Error ? error.constructor.name : 'Unknown'
      });
      
      // 🔄 FIRST FALLBACK: Try manual parsing of the error response
      try {
        logDebug(`🔄 [SAFE_GEN] First fallback: Attempting manual parsing of error response`);
        const errorResult = await this.handleGenerateObjectError<T>(error);
        logDebug(`✅ [SAFE_GEN] Manual parsing successful:`, { 
          model: model, 
          usage: errorResult.usage,
          tokensUsed: errorResult.usage.totalTokens
        });
        this.tokenTracker.trackUsage(model, errorResult.usage);
        return errorResult;

      } catch (parseError: unknown) {
        logWarning(`❌ [SAFE_GEN] Manual parsing failed:`, { 
          error: parseError instanceof Error ? parseError.message : String(parseError),
          errorType: parseError instanceof Error ? parseError.constructor.name : 'Unknown'
        });

        // 🔄 SECOND FALLBACK: Retry with remaining attempts
        if (numRetries > 0) {
          logWarning(`🔄 [SAFE_GEN] Second fallback: Retrying with ${numRetries - 1} attempts remaining`);
          return this.generateObject({
            model,
            schema,
            prompt,
            system,
            messages,
            numRetries: numRetries - 1
          });
        } else {
          // 🔄 THIRD FALLBACK: Try with fallback model and distilled schema
          logWarning(`🔄 [SAFE_GEN] Third fallback: Using fallback model with simplified schema`);
          try {
            let failedOutput = '';

            // Extract the failed output for better fallback processing
            if (NoObjectGeneratedError.isInstance(parseError)) {
              failedOutput = (parseError as any).text;
              // Find last `"url":` appear in the string, which is often the source of the problem
              failedOutput = failedOutput.slice(0, Math.min(failedOutput.lastIndexOf('"url":'), 8000));
              logDebug(`🔧 [SAFE_GEN] Extracted failed output for fallback processing:`, { 
                outputLength: failedOutput.length,
                outputPreview: failedOutput.substring(0, 200) + (failedOutput.length > 200 ? '...' : '')
              });
            }

            // Create a distilled version of the schema without descriptions
            const distilledSchema = this.createDistilledSchema(schema);
            logDebug(`🔧 [SAFE_GEN] Created simplified schema for fallback model`);

            const fallbackResult = await generateObject({
              model: getModel('fallback'),
              schema: distilledSchema,
              prompt: `Following the given JSON schema, extract the field from below: \n\n ${failedOutput}`,
              temperature: getToolConfig('fallback').temperature,
            });

            logDebug(`✅ [SAFE_GEN] Fallback model successful:`, { 
              model: 'fallback', 
              usage: fallbackResult.usage,
              tokensUsed: fallbackResult.usage.totalTokens
            });
            this.tokenTracker.trackUsage('fallback', fallbackResult.usage); // Track against fallback model
            logDebug('🔧 [SAFE_GEN] Simplified schema parsing successful!');
            return fallbackResult;
          } catch (fallbackError: unknown) {
            logWarning(`❌ [SAFE_GEN] Fallback model failed:`, { 
              error: fallbackError instanceof Error ? fallbackError.message : String(fallbackError),
              errorType: fallbackError instanceof Error ? fallbackError.constructor.name : 'Unknown'
            });
            
            // 🔄 FOURTH FALLBACK: Try parsing the fallback model's error response
            try {
              logDebug(`🔄 [SAFE_GEN] Fourth fallback: Manual parsing of fallback error response`);
              const lastChanceResult = await this.handleGenerateObjectError<T>(fallbackError);
              logDebug(`✅ [SAFE_GEN] Last chance parsing successful:`, { 
                model: 'fallback', 
                usage: lastChanceResult.usage,
                tokensUsed: lastChanceResult.usage.totalTokens
              });
              this.tokenTracker.trackUsage('fallback', lastChanceResult.usage);
              return lastChanceResult;
            } catch (finalError: unknown) {
              logError(`💀 [SAFE_GEN] All recovery mechanisms failed:`, { 
                originalError: error instanceof Error ? error.message : String(error),
                finalError: finalError instanceof Error ? finalError.message : String(finalError),
                originalErrorType: error instanceof Error ? error.constructor.name : 'Unknown',
                finalErrorType: finalError instanceof Error ? finalError.constructor.name : 'Unknown'
              });
              throw error; // Throw original error for better debugging
            }
          }
        }
      }
    }
  }

  /**
   * Handles NoObjectGeneratedError by attempting manual parsing
   * Tries both standard JSON parsing and Hjson parsing for maximum compatibility
   */
  private async handleGenerateObjectError<T>(error: unknown): Promise<GenerateObjectResult<T>> {
    if (NoObjectGeneratedError.isInstance(error)) {
      logWarning('🔧 [SAFE_GEN] Object not generated according to schema, attempting manual parsing', { 
        errorText: (error as any).text?.substring(0, 200) + '...' 
      });
      
      try {
        // 🎯 FIRST ATTEMPT: Standard JSON parsing
        logDebug(`🔧 [SAFE_GEN] Attempting standard JSON parsing`);
        const partialResponse = JSON.parse((error as any).text);
        logDebug(`✅ [SAFE_GEN] Standard JSON parse success!`);
        return {
          object: partialResponse as T,
          usage: (error as any).usage
        };
      } catch (parseError) {
        logWarning(`❌ [SAFE_GEN] Standard JSON parsing failed:`, { 
          error: parseError instanceof Error ? parseError.message : String(parseError) 
        });
        
        // 🔄 SECOND ATTEMPT: Use Hjson for more lenient parsing
        try {
          logDebug(`🔧 [SAFE_GEN] Attempting Hjson parsing for more lenient parsing`);
          const hjsonResponse = Hjson.parse((error as any).text);
          logDebug(`✅ [SAFE_GEN] Hjson parse success!`);
          return {
            object: hjsonResponse as T,
            usage: (error as any).usage
          };
        } catch (hjsonError) {
          logError('❌ [SAFE_GEN] Both JSON and Hjson parsing failed:', { 
            jsonError: parseError instanceof Error ? parseError.message : String(parseError),
            hjsonError: hjsonError instanceof Error ? hjsonError.message : String(hjsonError)
          });
          throw error;
        }
      }
    }
    throw error;
  }
}