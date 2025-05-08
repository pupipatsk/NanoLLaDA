const summarize = async (input: string) => {
    const response = await fetch(process.env.NEXT_PUBLIC_API_URL!, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: 'สรุปข้อความต่อไปนี้:\n' + input })
    });

    if (!response.ok) throw new Error('Failed to get summary');
    const data = await response.json();
    return data['summary'] as string;
}

/**
 * @see https://grok.com/share/bGVnYWN5_030d8fcd-3ce6-460b-a8da-f66f7244b73a
 */
export class LangChainSummarizer {
    private readonly chunkSize: number = 1000;
    private readonly minChunkSize: number = 500;

    // Splits text into chunks of ~1000 characters, ensuring no chunk is < 500 characters
    private splitText(text: string): string[] {
        if (text.length <= this.chunkSize) return [text];

        const chunks: string[] = [];
        let start = 0;
        const sentences = text.match(/[^.!?]+[.!?]+|\S+/g) || [text]; // Split by sentences or fallback to whole text

        while (start < sentences.length) {
            let chunk = '';
            let sentenceIndex = start;

            // Accumulate sentences until chunk is ~1000 characters but not < 500
            while (
                sentenceIndex < sentences.length &&
                chunk.length < this.chunkSize &&
                (chunk.length < this.minChunkSize || chunk.length + (sentences[sentenceIndex]?.length || 0) <= this.chunkSize)
            ) {
                chunk += (sentences[sentenceIndex] || '') + ' ';
                sentenceIndex++;
            }

            // If chunk is too small and not the last, merge with next sentences
            if (chunk.length < this.minChunkSize && sentenceIndex < sentences.length) {
                while (
                    sentenceIndex < sentences.length &&
                    chunk.length < this.minChunkSize
                ) {
                    chunk += (sentences[sentenceIndex] || '') + ' ';
                    sentenceIndex++;
                }
            }

            if (chunk.trim()) chunks.push(chunk.trim());
            start = sentenceIndex;
        }

        return chunks;
    }

    // Map step: Summarize each chunk
    private async mapSummaries(chunks: string[]): Promise<string[]> {
        const summaries = await Promise.all(
            chunks.map((chunk) => summarize(chunk))
        );
        return summaries.filter((summary) => summary.trim());
    }

    // Reduce step: Combine and summarize the summaries
    private async reduceSummaries(summaries: string[]): Promise<string> {
        if (summaries.length === 0) return '';
        if (summaries.length === 1) return summaries[0];

        const combinedSummaries = summaries.join(' ');
        if (combinedSummaries.length <= this.chunkSize) {
            try {
                return await summarize(combinedSummaries);
            } catch (error) {
                console.error('Error in reduce step:', error);
                return combinedSummaries; // Fallback to combined summaries
            }
        }

        // If combined summaries are too long, recursively split and summarize
        const subChunks = this.splitText(combinedSummaries);
        const subSummaries = await this.mapSummaries(subChunks);
        return this.reduceSummaries(subSummaries);
    }

    // Public method to summarize input text
    public async summarizeText(input: string): Promise<string> {
        if (!input.trim()) return '';

        const chunks = this.splitText(input);
        const summaries = await this.mapSummaries(chunks);
        return this.reduceSummaries(summaries);
    }
}