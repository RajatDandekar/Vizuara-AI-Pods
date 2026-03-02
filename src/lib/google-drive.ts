/**
 * Google Drive API wrapper using googleapis.
 * Replaces rclone CLI calls for Vercel compatibility.
 */
import { google } from 'googleapis';
import { Readable } from 'stream';

// Parent folder with subfolders: "notebooks" and "case-studies"
const DRIVE_NOTEBOOKS_FOLDER_ID = '1TuuzIfYRkd7mtQZnGp9DtyMsa12xCmrn';
const DRIVE_CASE_STUDIES_FOLDER_ID = '18KGh09A-_dO9Aek1Nl0UUJSv7Ra8dlP8';

function getDriveClient() {
  const keyJson = process.env.GOOGLE_SERVICE_ACCOUNT_KEY;
  if (!keyJson) {
    throw new Error('GOOGLE_SERVICE_ACCOUNT_KEY env var not set');
  }

  const credentials = JSON.parse(keyJson);
  const auth = new google.auth.GoogleAuth({
    credentials,
    scopes: ['https://www.googleapis.com/auth/drive.file'],
  });

  return google.drive({ version: 'v3', auth });
}

/**
 * Find or create a subfolder inside the root Drive folder.
 * Returns the folder ID. Used to organize notebooks by pod slug.
 */
async function getOrCreateSubfolder(
  drive: ReturnType<typeof google.drive>,
  parentId: string,
  folderName: string
): Promise<string> {
  // Check if subfolder already exists
  const query = `'${parentId}' in parents and name='${folderName}' and mimeType='application/vnd.google-apps.folder' and trashed=false`;
  const existing = await drive.files.list({
    q: query,
    fields: 'files(id)',
    pageSize: 1,
  });

  if (existing.data.files && existing.data.files.length > 0) {
    return existing.data.files[0].id!;
  }

  // Create the subfolder
  const folder = await drive.files.create({
    requestBody: {
      name: folderName,
      mimeType: 'application/vnd.google-apps.folder',
      parents: [parentId],
    },
    fields: 'id',
  });

  return folder.data.id!;
}

/**
 * Upload a file to the shared Drive folder.
 * - type 'notebook' → notebooks/{subfolder}/filename
 * - type 'case-study' → case-studies/{subfolder}/filename
 * subfolder is typically the pod slug (e.g., 'grpo', 'world-models').
 * Returns the file ID and a direct CDN URL.
 */
export async function uploadToDrive(
  filename: string,
  content: Buffer,
  mimeType: string,
  subfolder?: string,
  type: 'notebook' | 'case-study' = 'notebook'
): Promise<{ fileId: string; url: string }> {
  const drive = getDriveClient();

  const rootId = type === 'case-study' ? DRIVE_CASE_STUDIES_FOLDER_ID : DRIVE_NOTEBOOKS_FOLDER_ID;
  let parentId = rootId;
  if (subfolder) {
    parentId = await getOrCreateSubfolder(drive, rootId, subfolder);
  }

  const response = await drive.files.create({
    requestBody: {
      name: filename,
      parents: [parentId],
    },
    media: {
      mimeType,
      body: Readable.from(content),
    },
    fields: 'id',
  });

  const fileId = response.data.id!;

  // Make the file accessible via link
  await drive.permissions.create({
    fileId,
    requestBody: {
      role: 'reader',
      type: 'anyone',
    },
  });

  const url = `https://lh3.googleusercontent.com/d/${fileId}=w2000`;

  return { fileId, url };
}

/**
 * Delete a file from Google Drive by its file ID.
 */
export async function deleteFromDrive(fileId: string): Promise<void> {
  const drive = getDriveClient();
  await drive.files.delete({ fileId });
}

/**
 * Get Colab URL from a Drive file ID.
 */
export function getColabUrl(fileId: string): string {
  return `https://colab.research.google.com/drive/${fileId}`;
}
