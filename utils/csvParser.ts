import { DataRow } from '../types';
import { logSyncFunctionIO } from './logger';

const _parseCSV = (content: string): DataRow[] => {
  const lines = content.trim().split('\n');
  if (lines.length < 2) return [];

  // Auto-detect delimiter based on the first line (header)
  const detectDelimiter = (text: string): string => {
    const candidates = [',', ';', '\t', '|'];
    let bestDelimiter = ',';
    let maxCount = 0;

    candidates.forEach(delim => {
      const count = text.split(delim).length - 1;
      if (count > maxCount) {
        maxCount = count;
        bestDelimiter = delim;
      }
    });

    return bestDelimiter;
  };

  const delimiter = detectDelimiter(lines[0]);

  // Robust CSV line splitter that respects quoted strings containing the delimiter
  const splitLine = (text: string, delim: string): string[] => {
    const result: string[] = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === delim && !inQuotes) {
        result.push(current.trim().replace(/^"|"$/g, '')); // Remove surrounding quotes
        current = '';
      } else {
        current += char;
      }
    }
    result.push(current.trim().replace(/^"|"$/g, ''));
    return result;
  };

  const headers = splitLine(lines[0], delimiter);

  const data: DataRow[] = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    const values = splitLine(line, delimiter);

    // Ensure row matches header length roughly to avoid malformed rows
    if (values.length !== headers.length) {
      // console.warn('Skipping malformed row', values);
      continue;
    }

    const row: DataRow = {};
    headers.forEach((header, index) => {
      let val = values[index];
      if (val === undefined) val = '';

      if (val.trim() === '') {
        row[header] = val;
      } else {
        // Use Number() for strict parsing. 
        // parseFloat() is too aggressive (e.g. "2023-01-05" -> 2023).
        // Number("2023-01-05") -> NaN.
        // Handle commas in numbers (e.g. "1,000") by stripping them before checking.
        const cleanVal = val.replace(/,/g, '');
        const num = Number(cleanVal);

        // Check if it's a valid number and not an empty string after cleaning
        if (!isNaN(num) && cleanVal.trim() !== '') {
          // Check for IDs that start with 0 (e.g. "0123") - keep as string
          // However, strictly "0" or "0.5" are numbers.
          if (val.trim().startsWith('0') && val.trim() !== '0' && !val.trim().startsWith('0.')) {
            row[header] = val;
          } else {
            row[header] = num;
          }
        } else {
          row[header] = val;
        }
      }
    });
    data.push(row);
  }
  return data;
};

export const parseCSV = logSyncFunctionIO('parseCSV', _parseCSV);