/**
 * Zip creation utility using archiver.
 * Replaces `zip` CLI calls for Vercel compatibility.
 */
import archiver from 'archiver';

/**
 * Create a zip buffer from an array of named buffers.
 */
export async function createZipFromBuffers(
  files: Array<{ name: string; buffer: Buffer }>
): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const archive = archiver('zip', { zlib: { level: 6 } });
    const chunks: Buffer[] = [];

    archive.on('data', (chunk: Buffer) => chunks.push(chunk));
    archive.on('end', () => resolve(Buffer.concat(chunks)));
    archive.on('error', reject);

    for (const file of files) {
      archive.append(file.buffer, { name: file.name });
    }

    archive.finalize();
  });
}
