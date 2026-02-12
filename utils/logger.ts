/**
 * Structured logging utility for the Finance Forecasting frontend.
 *
 * Environment-aware: only logs in development mode (Vite DEV).
 * Provides truncation for large objects to prevent console flooding.
 */

const isDev = import.meta.env.DEV;

const MAX_STR_LENGTH = 500;
const MAX_ARRAY_ITEMS = 5;
const MAX_OBJECT_KEYS = 10;

function truncateValue(value: unknown, depth = 0): unknown {
  if (depth > 2) return `<${typeof value}>`;
  if (value === null || value === undefined) return value;

  if (typeof value === 'string') {
    return value.length > MAX_STR_LENGTH
      ? `[string len=${value.length}] "${value.slice(0, MAX_STR_LENGTH)}..."`
      : value;
  }

  if (value instanceof Blob || value instanceof File) {
    return `${value.constructor.name}(size=${value.size}, type=${value.type})`;
  }

  if (Array.isArray(value)) {
    if (value.length > MAX_ARRAY_ITEMS) {
      return {
        __truncated: true,
        length: value.length,
        sample: value.slice(0, MAX_ARRAY_ITEMS).map((v) => truncateValue(v, depth + 1)),
      };
    }
    return value.map((v) => truncateValue(v, depth + 1));
  }

  if (typeof value === 'object') {
    const keys = Object.keys(value as Record<string, unknown>);
    if (keys.length > MAX_OBJECT_KEYS) {
      const sample: Record<string, unknown> = {};
      keys.slice(0, MAX_OBJECT_KEYS).forEach((k) => {
        sample[k] = truncateValue((value as Record<string, unknown>)[k], depth + 1);
      });
      return { __truncated: true, totalKeys: keys.length, sample };
    }
    const result: Record<string, unknown> = {};
    keys.forEach((k) => {
      result[k] = truncateValue((value as Record<string, unknown>)[k], depth + 1);
    });
    return result;
  }

  return value;
}

export const logger = {
  debug: (message: string, ...args: unknown[]) => {
    if (isDev) console.debug(`[DEBUG] ${message}`, ...args.map((a) => truncateValue(a)));
  },
  info: (message: string, ...args: unknown[]) => {
    if (isDev) console.info(`[INFO] ${message}`, ...args.map((a) => truncateValue(a)));
  },
  warn: (message: string, ...args: unknown[]) => {
    console.warn(`[WARN] ${message}`, ...args.map((a) => truncateValue(a)));
  },
  error: (message: string, ...args: unknown[]) => {
    console.error(`[ERROR] ${message}`, ...args.map((a) => truncateValue(a)));
  },
};

/**
 * Wraps an async service function to log its inputs and outputs at DEBUG level.
 *
 * In production builds, returns the original function unchanged (zero overhead).
 */
export function logFunctionIO<T extends (...args: any[]) => Promise<any>>(
  name: string,
  fn: T,
): T {
  if (!isDev) return fn;

  const wrapped = async (...args: Parameters<T>): Promise<ReturnType<T>> => {
    logger.debug(`[ENTER] ${name}`, ...args);
    const start = performance.now();
    try {
      const result = await fn(...args);
      const elapsed = (performance.now() - start).toFixed(1);
      logger.debug(`[EXIT]  ${name} (${elapsed}ms)`, result);
      return result;
    } catch (error) {
      const elapsed = (performance.now() - start).toFixed(1);
      logger.error(`[ERROR] ${name} (${elapsed}ms)`, error);
      throw error;
    }
  };
  return wrapped as T;
}

/**
 * Wraps a sync function to log its inputs and outputs at DEBUG level.
 *
 * In production builds, returns the original function unchanged (zero overhead).
 */
export function logSyncFunctionIO<T extends (...args: any[]) => any>(
  name: string,
  fn: T,
): T {
  if (!isDev) return fn;

  const wrapped = (...args: Parameters<T>): ReturnType<T> => {
    logger.debug(`[ENTER] ${name}`, ...args);
    const start = performance.now();
    try {
      const result = fn(...args);
      const elapsed = (performance.now() - start).toFixed(1);
      logger.debug(`[EXIT]  ${name} (${elapsed}ms)`, result);
      return result;
    } catch (error) {
      const elapsed = (performance.now() - start).toFixed(1);
      logger.error(`[ERROR] ${name} (${elapsed}ms)`, error);
      throw error;
    }
  };
  return wrapped as T;
}
