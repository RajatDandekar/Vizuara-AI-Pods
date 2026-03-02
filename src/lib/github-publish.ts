/**
 * GitHub publish module using Octokit.
 * Creates atomic multi-file commits via the Git Tree API.
 *
 * Replaces local filesystem copies + manual git push for Vercel deployment.
 */
import { Octokit } from 'octokit';

const OWNER = 'RajatDandekar';
const REPO = 'Vizuara-AI-Pods';
const BRANCH = 'main';

function getOctokit(): Octokit {
  const token = process.env.GITHUB_TOKEN;
  if (!token) throw new Error('GITHUB_TOKEN env var not set');
  return new Octokit({ auth: token });
}

export interface FileToCommit {
  path: string; // relative to repo root, e.g. "content/courses/my-course/pods/my-pod/pod.json"
  content: string | Buffer; // string for text, Buffer for binary
  encoding?: 'utf-8' | 'base64';
}

/**
 * Read a file from the repo. Returns null if not found.
 */
export async function getRepoFileContent(
  filePath: string
): Promise<string | null> {
  const octokit = getOctokit();

  try {
    const response = await octokit.rest.repos.getContent({
      owner: OWNER,
      repo: REPO,
      path: filePath,
      ref: BRANCH,
    });

    const data = response.data;
    if ('content' in data && data.content) {
      return Buffer.from(data.content, 'base64').toString('utf-8');
    }
    return null;
  } catch (err: unknown) {
    if (err && typeof err === 'object' && 'status' in err && err.status === 404) {
      return null;
    }
    throw err;
  }
}

/**
 * Create an atomic commit with multiple files using the Git Tree API.
 *
 * This creates a single commit with all files at once, avoiding partial
 * states if a multi-file push were to fail halfway.
 *
 * Returns the commit SHA and URL.
 */
export async function commitAndPush(
  files: FileToCommit[],
  message: string
): Promise<{ sha: string; url: string }> {
  const octokit = getOctokit();

  // 1. Get the latest commit SHA on the branch
  const { data: ref } = await octokit.rest.git.getRef({
    owner: OWNER,
    repo: REPO,
    ref: `heads/${BRANCH}`,
  });
  const latestCommitSha = ref.object.sha;

  // 2. Get the tree SHA of the latest commit
  const { data: latestCommit } = await octokit.rest.git.getCommit({
    owner: OWNER,
    repo: REPO,
    commit_sha: latestCommitSha,
  });
  const baseTreeSha = latestCommit.tree.sha;

  // 3. Create blobs for each file
  const treeItems: Array<{
    path: string;
    mode: '100644';
    type: 'blob';
    sha: string;
  }> = [];

  for (const file of files) {
    let content: string;
    let encoding: 'utf-8' | 'base64';

    if (Buffer.isBuffer(file.content)) {
      content = file.content.toString('base64');
      encoding = 'base64';
    } else {
      content = file.content;
      encoding = file.encoding || 'utf-8';
    }

    const { data: blob } = await octokit.rest.git.createBlob({
      owner: OWNER,
      repo: REPO,
      content,
      encoding,
    });

    treeItems.push({
      path: file.path,
      mode: '100644',
      type: 'blob',
      sha: blob.sha,
    });
  }

  // 4. Create a new tree with the base tree + new files
  const { data: newTree } = await octokit.rest.git.createTree({
    owner: OWNER,
    repo: REPO,
    base_tree: baseTreeSha,
    tree: treeItems,
  });

  // 5. Create the commit
  const { data: newCommit } = await octokit.rest.git.createCommit({
    owner: OWNER,
    repo: REPO,
    message,
    tree: newTree.sha,
    parents: [latestCommitSha],
  });

  // 6. Update the branch ref to point to the new commit
  await octokit.rest.git.updateRef({
    owner: OWNER,
    repo: REPO,
    ref: `heads/${BRANCH}`,
    sha: newCommit.sha,
  });

  return {
    sha: newCommit.sha,
    url: newCommit.html_url,
  };
}
