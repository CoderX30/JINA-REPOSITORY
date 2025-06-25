import fs from 'fs';
import { extractNgrams } from '../utils/text-tools';

async function main() {
    const args = process.argv.slice(2);
    if (args.length === 0) {
        console.error('Please provide a file path');
        process.exit(1);
    }

    const filePath = args[0];
    const n = parseInt(args[1]) || 3; // Default to 3-grams
    const minFreq = parseInt(args[2]) || 2; // Default minimum frequency of 2
    const minPMI = parseFloat(args[3]) || 1.0; // Default minimum PMI of 1.0

    try {
        const text = await fs.promises.readFile(filePath, 'utf-8');
        const results = extractNgrams(text, n, minFreq, minPMI);

        console.log('\nN-gram Analysis Results:');
        console.log('------------------------');
        results.forEach(({ ngram, freq, pmi }) => {
            if (pmi !== undefined) {
                console.log(`${ngram}: ${freq} (PMI: ${pmi.toFixed(2)})`);
            } else {
                console.log(`${ngram}: ${freq}`);
            }
        });
    } catch (err) {
        const error = err as Error;
        console.error('Error:', error.message);
        process.exit(1);
    }
}

main(); 