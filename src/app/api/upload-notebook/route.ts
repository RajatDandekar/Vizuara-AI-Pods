import { NextRequest, NextResponse } from 'next/server';
import { writeFile, unlink } from 'fs/promises';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { tmpdir } from 'os';
import { join } from 'path';

const execFileAsync = promisify(execFile);

const RCLONE = '/Users/raj/.local/bin/rclone';
const REMOTE = 'gdrive:/';
const FOLDER_ID = '1ZbfI4AU_DbqeHSKkbVX28DO-Pr4_PPI6';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  try {
    const { notebook, filename } = await request.json();

    if (!notebook || !filename) {
      return NextResponse.json(
        { error: 'notebook and filename are required' },
        { status: 400 }
      );
    }

    // Write notebook to a temp file
    const tmpPath = join(tmpdir(), filename);
    await writeFile(tmpPath, JSON.stringify(notebook, null, 2));

    try {
      // Upload to Google Drive via rclone
      await execFileAsync(RCLONE, [
        'copy', tmpPath, REMOTE,
        '--drive-root-folder-id', FOLDER_ID, '-v',
      ]);

      // Get the Drive file ID via rclone link
      const { stdout } = await execFileAsync(RCLONE, [
        'link', `${REMOTE}${filename}`,
        '--drive-root-folder-id', FOLDER_ID,
      ]);

      // Extract file ID from the link URL
      // rclone link returns: https://drive.google.com/open?id=FILE_ID
      const match = stdout.trim().match(/id=([a-zA-Z0-9_-]+)/);
      if (!match) {
        return NextResponse.json(
          { error: 'Failed to get Drive file ID' },
          { status: 500 }
        );
      }

      const fileId = match[1];
      const colabUrl = `https://colab.research.google.com/drive/${fileId}`;

      return NextResponse.json({ colabUrl, fileId });
    } finally {
      // Clean up temp file
      await unlink(tmpPath).catch(() => {});
    }
  } catch (error) {
    console.error('Upload notebook error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Upload failed' },
      { status: 500 }
    );
  }
}
