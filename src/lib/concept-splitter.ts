/**
 * Splits an article draft on ## headings and extracts concepts for notebook generation.
 * TypeScript port of concept_splitter.py
 */

export interface ConceptPlanOutput {
  index: number;
  concept_name: string;
  notebook_filename: string;
  description: string;
  subsections: string[];
  key_terms: string[];
  equations: string[];
  code_blocks: Array<{ language: string; code: string }>;
  figure_descriptions: string[];
  prerequisites: string[];
  suggested_final_output: string;
  word_count: number;
}

interface Section {
  title: string;
  article_title?: string;
  raw_content: string;
}

function extractSections(content: string): Section[] {
  const sections: Section[] = [];
  const parts = content.split('\n## ');

  // First part is everything before the first ##
  if (parts[0].trim()) {
    const titleMatch = parts[0].match(/^# (.+)$/m);
    const title = titleMatch ? titleMatch[1].trim() : 'Introduction';
    sections.push({
      title: 'Introduction',
      article_title: title,
      raw_content: parts[0],
    });
  }

  // Remaining parts are ## sections
  for (let i = 1; i < parts.length; i++) {
    const lines = parts[i].split('\n');
    const sectionTitle = lines[0].trim();
    const sectionContent = lines.slice(1).join('\n');
    sections.push({
      title: sectionTitle,
      raw_content: `## ${sectionTitle}\n${sectionContent}`,
    });
  }

  return sections;
}

function extractEquations(content: string): string[] {
  const equations: string[] = [];

  // Block equations: {{EQUATION: ...}}
  for (const match of content.matchAll(/\{\{EQUATION:\s*([\s\S]*?)\}\}/g)) {
    equations.push(match[1].trim());
  }

  // Display equations: $$...$$
  for (const match of content.matchAll(/\$\$([\s\S]*?)\$\$/g)) {
    const eq = match[1].trim();
    if (eq) equations.push(eq);
  }

  // Inline equations: $...$
  for (const match of content.matchAll(/(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)/g)) {
    equations.push(match[1].trim());
  }

  return equations;
}

function extractCodeBlocks(content: string): Array<{ language: string; code: string }> {
  const blocks: Array<{ language: string; code: string }> = [];

  for (const match of content.matchAll(/```(\w*)\n([\s\S]*?)```/g)) {
    blocks.push({
      language: match[1] || 'python',
      code: match[2].trim(),
    });
  }

  return blocks;
}

function extractFigures(content: string): string[] {
  const figures: string[] = [];
  for (const match of content.matchAll(/\{\{FIGURE:\s*([\s\S]*?)\}\}/g)) {
    figures.push(match[1].trim());
  }
  return figures;
}

function extractKeyTerms(content: string): string[] {
  const terms = new Set<string>();
  for (const match of content.matchAll(/\*\*([^*]+)\*\*/g)) {
    const term = match[1].trim();
    if (term.split(/\s+/).length <= 6) {
      terms.add(term);
    }
  }
  return Array.from(terms);
}

function extractFirstParagraph(content: string): string {
  const lines = content.split('\n');
  const paraLines: string[] = [];
  let started = false;

  for (const line of lines) {
    if (!started) {
      if (line.trim() && !line.startsWith('#') && !line.startsWith('---')) {
        started = true;
        paraLines.push(line.trim());
      }
    } else {
      if (!line.trim() || line.startsWith('#') || line.startsWith('---')) break;
      paraLines.push(line.trim());
    }
  }

  return paraLines.join(' ').slice(0, 500);
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '')
    .replace(/[\s-]+/g, '_')
    .replace(/^_|_$/g, '')
    .slice(0, 50);
}

function determineNotebookOutput(title: string, content: string): string {
  const titleLower = title.toLowerCase();
  const contentLower = content.toLowerCase();

  if (titleLower.includes('world model') || contentLower.includes('dream') || contentLower.includes('v-m-c'))
    return 'Agent playing a game inside its own learned dream world, with side-by-side real vs dreamed frames';
  if (titleLower.includes('dreamer') || contentLower.includes('rssm'))
    return 'Trained RSSM world model predicting future frames, with imagination rollout visualization';
  if (titleLower.includes('jepa'))
    return 'Visualization of I-JEPA predicting masked patch representations, with t-SNE of learned embeddings';
  if (titleLower.includes('genie') || titleLower.includes('interactive world'))
    return 'Simple interactive environment generated from a single image, responding to user actions';
  if (titleLower.includes('vla') || contentLower.includes('vision-language-action') || contentLower.includes('π0'))
    return 'Simulated robot arm executing a language-commanded pick-and-place task';
  if (contentLower.includes('diffusion'))
    return 'Grid of images generated from pure noise through the learned reverse diffusion process';
  if (contentLower.includes('attention'))
    return 'Attention weight heatmap visualization on real input sequences';
  if (contentLower.includes('vae') || contentLower.includes('autoencoder'))
    return 'Latent space interpolation showing smooth transitions between generated outputs';

  return 'Trained model producing visible outputs demonstrating the core concept';
}

/**
 * Split an article draft into notebook concepts.
 * Returns the concept plan array ready for ConceptPlan transformation.
 */
export function splitArticleIntoConcepts(
  articleContent: string,
  minWords: number = 100
): ConceptPlanOutput[] {
  const sections = extractSections(articleContent);
  const concepts: ConceptPlanOutput[] = [];

  const skipPatterns = ['references', 'big picture', 'the convergence'];
  let conceptIndex = 0;

  for (const section of sections) {
    const { title, raw_content: content } = section;

    // Skip short sections
    if (content.split(/\s+/).length < minWords) continue;

    // Skip reference/conclusion sections
    if (skipPatterns.some((p) => title.toLowerCase().includes(p))) continue;

    conceptIndex++;

    const prerequisites = concepts.map((c) => c.concept_name);
    const subsections = Array.from(content.matchAll(/### (.+)/g)).map((m) => m[1]);

    concepts.push({
      index: conceptIndex,
      concept_name: title,
      notebook_filename: `${String(conceptIndex).padStart(2, '0')}_${slugify(title)}.ipynb`,
      description: extractFirstParagraph(content),
      subsections,
      key_terms: extractKeyTerms(content),
      equations: extractEquations(content),
      code_blocks: extractCodeBlocks(content),
      figure_descriptions: extractFigures(content),
      prerequisites,
      suggested_final_output: determineNotebookOutput(title, content),
      word_count: content.split(/\s+/).length,
    });
  }

  return concepts;
}
