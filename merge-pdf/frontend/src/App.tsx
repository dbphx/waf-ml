import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";
import type { DrivePreviewFile, Job, UploadReviewFile, User } from "./types";

const apiBaseURL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8080/api";
const tokenStorageKey = "merge-pdf-token";

function App() {
  const [token, setToken] = useState<string>(() => localStorage.getItem(tokenStorageKey) ?? "");
  const [user, setUser] = useState<User | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [activeTab, setActiveTab] = useState<"drive" | "upload" | "history">("drive");
  const [loading, setLoading] = useState(false);

  const [loginEmail, setLoginEmail] = useState("user@example.com");
  const [loginPassword, setLoginPassword] = useState("ChangeMe123!");

  const [driveURL, setDriveURL] = useState("");
  const [driveFiles, setDriveFiles] = useState<DrivePreviewFile[]>([]);

  const [uploadFiles, setUploadFiles] = useState<UploadReviewFile[]>([]);

  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);

  useEffect(() => {
    if (!token) {
      setUser(null);
      setJobs([]);
      setSelectedJob(null);
      return;
    }

    void bootstrap();
  }, [token]);

  const sortedUploadFiles = useMemo(
    () => [...uploadFiles].sort((a, b) => a.order - b.order || a.file.name.localeCompare(b.file.name)),
    [uploadFiles]
  );

  async function bootstrap() {
    try {
      const currentUser = await api<User>("/me", { token });
      setUser(currentUser);
      const payload = await api<{ jobs: Job[] }>("/jobs", { token });
      setJobs(payload.jobs);
    } catch (err) {
      localStorage.removeItem(tokenStorageKey);
      setToken("");
      setError(getErrorMessage(err));
    }
  }

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const payload = await api<{ token: string; user: User }>("/auth/login", {
        method: "POST",
        body: JSON.stringify({ email: loginEmail, password: loginPassword })
      });
      localStorage.setItem(tokenStorageKey, payload.token);
      setToken(payload.token);
      setUser(payload.user);
      setNotice("Logged in successfully.");
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  async function handleLogout() {
    localStorage.removeItem(tokenStorageKey);
    setToken("");
    setUser(null);
    setNotice("Logged out.");
  }

  async function handleDrivePreview() {
    setLoading(true);
    setError("");
    try {
      const payload = await api<{ files: DrivePreviewFile[] }>("/drive/preview", {
        method: "POST",
        token,
        body: JSON.stringify({ url: driveURL })
      });
      setDriveFiles(payload.files);
      setNotice(`Loaded ${payload.files.length} PDF file(s) from Drive.`);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  async function handleDriveMerge() {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${apiBaseURL}/merge/drive`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ url: driveURL })
      });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      await downloadResponse(response, "drive-merged.pdf");
      setNotice("Drive merge completed.");
      await refreshJobs();
      setActiveTab("history");
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  async function handleUploadMerge() {
    setLoading(true);
    setError("");
    try {
      const formData = new FormData();
      const orders: Record<string, number> = {};
      sortedUploadFiles.forEach((item) => {
        formData.append("files", item.file);
        orders[item.file.name] = item.order;
      });
      formData.append("orders", JSON.stringify(orders));

      const response = await fetch(`${apiBaseURL}/merge/upload`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`
        },
        body: formData
      });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      await downloadResponse(response, "upload-merged.pdf");
      setNotice("Upload merge completed.");
      setUploadFiles([]);
      await refreshJobs();
      setActiveTab("history");
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  async function refreshJobs() {
    const payload = await api<{ jobs: Job[] }>("/jobs", { token });
    setJobs(payload.jobs);
  }

  async function viewJob(jobId: number) {
    try {
      const payload = await api<Job>(`/jobs/${jobId}`, { token });
      setSelectedJob(payload);
    } catch (err) {
      setError(getErrorMessage(err));
    }
  }

  async function downloadJob(jobId: number, fileName: string) {
    try {
      const response = await fetch(`${apiBaseURL}/jobs/${jobId}/download`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      await downloadResponse(response, fileName);
    } catch (err) {
      setError(getErrorMessage(err));
    }
  }

  async function deleteJob(jobId: number) {
    try {
      await api(`/jobs/${jobId}`, { method: "DELETE", token });
      if (selectedJob?.id === jobId) {
        setSelectedJob(null);
      }
      await refreshJobs();
      setNotice("Job deleted.");
    } catch (err) {
      setError(getErrorMessage(err));
    }
  }

  function onFilesSelected(event: ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files ?? []).filter((file) => file.name.toLowerCase().endsWith(".pdf"));
    setUploadFiles(
      selected.map((file, index) => ({
        file,
        order: index + 1
      }))
    );
    setNotice(`Prepared ${selected.length} local PDF file(s).`);
  }

  function updateUploadOrder(name: string, order: number) {
    setUploadFiles((current) =>
      current.map((item) => (item.file.name === name ? { ...item, order } : item))
    );
  }

  function removeUploadFile(name: string) {
    setUploadFiles((current) => current.filter((item) => item.file.name !== name));
  }

  if (!token || !user) {
    return (
      <main className="shell shell-login">
        <section className="hero-card">
          <p className="eyebrow">Authenticated PDF workspace</p>
          <h1>Merge Drive folders or local PDFs without losing history.</h1>
          <p className="hero-copy">
            The app stores merged outputs in MinIO, keeps job history per account, and enforces Drive ordering from numeric prefixes in filenames.
          </p>
        </section>

        <section className="panel login-panel">
          <form onSubmit={handleLogin}>
            <label>
              Email
              <input value={loginEmail} onChange={(event) => setLoginEmail(event.target.value)} type="email" required />
            </label>
            <label>
              Password
              <input value={loginPassword} onChange={(event) => setLoginPassword(event.target.value)} type="password" required />
            </label>
            <button type="submit" disabled={loading}>
              {loading ? "Signing in..." : "Sign in"}
            </button>
          </form>
          <p className="hint">Seed accounts default to `user@example.com` / `ChangeMe123!`.</p>
          {error ? <p className="feedback error">{error}</p> : null}
          {notice ? <p className="feedback success">{notice}</p> : null}
        </section>
      </main>
    );
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Merge PDF control room</p>
          <h1>{user.email}</h1>
        </div>
        <div className="topbar-actions">
          <span className="role-pill">{user.role}</span>
          <button className="ghost" onClick={handleLogout}>
            Log out
          </button>
        </div>
      </header>

      <nav className="tabs">
        <button className={activeTab === "drive" ? "active" : ""} onClick={() => setActiveTab("drive")}>Drive Link</button>
        <button className={activeTab === "upload" ? "active" : ""} onClick={() => setActiveTab("upload")}>Upload Files</button>
        <button className={activeTab === "history" ? "active" : ""} onClick={() => setActiveTab("history")}>History</button>
      </nav>

      {error ? <p className="feedback error">{error}</p> : null}
      {notice ? <p className="feedback success">{notice}</p> : null}

      {activeTab === "drive" ? (
        <section className="panel grid">
          <div className="stack">
            <label>
              Google Drive folder link
              <input
                value={driveURL}
                onChange={(event) => setDriveURL(event.target.value)}
                placeholder="https://drive.google.com/drive/u/0/folders/..."
              />
            </label>
            <div className="actions">
              <button onClick={handleDrivePreview} disabled={loading || !driveURL}>
                Preview Files
              </button>
              <button className="ghost" onClick={handleDriveMerge} disabled={loading || !driveFiles.length}>
                Merge by Filename Numbers
              </button>
            </div>
          </div>

          <div className="panel inset">
            <h2>Drive preview</h2>
            <table>
              <thead>
                <tr>
                  <th>Order</th>
                  <th>Name</th>
                  <th>Size</th>
                </tr>
              </thead>
              <tbody>
                {driveFiles.length ? (
                  driveFiles.map((file) => (
                    <tr key={file.sourceId}>
                      <td>{file.extractedOrder}</td>
                      <td>
                        <a href={file.webViewLink} target="_blank" rel="noreferrer">
                          {file.name}
                        </a>
                      </td>
                      <td>{formatBytes(file.size)}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={3}>Paste a shared folder link, then preview its PDF files.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {activeTab === "upload" ? (
        <section className="panel grid">
          <div className="stack">
            <label className="upload-dropzone">
              <span>Choose PDF files</span>
              <input type="file" accept="application/pdf" multiple onChange={onFilesSelected} />
            </label>
            <button onClick={handleUploadMerge} disabled={loading || !uploadFiles.length}>
              Merge Uploaded Files
            </button>
          </div>
          <div className="panel inset">
            <h2>Upload review</h2>
            <table>
              <thead>
                <tr>
                  <th>Order</th>
                  <th>Name</th>
                  <th>Size</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {sortedUploadFiles.length ? (
                  sortedUploadFiles.map((item) => (
                    <tr key={item.file.name}>
                      <td>
                        <input
                          className="order-input"
                          type="number"
                          min={1}
                          value={item.order}
                          onChange={(event) => updateUploadOrder(item.file.name, Number(event.target.value))}
                        />
                      </td>
                      <td>{item.file.name}</td>
                      <td>{formatBytes(item.file.size)}</td>
                      <td>
                        <button className="ghost compact" onClick={() => removeUploadFile(item.file.name)}>
                          Remove
                        </button>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={4}>Upload one or more PDF files to start a merge job.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {activeTab === "history" ? (
        <section className="history-layout">
          <div className="panel">
            <h2>Job history</h2>
            <ul className="history-list">
              {jobs.length ? (
                jobs.map((job) => (
                  <li key={job.id}>
                    <button className="history-card" onClick={() => viewJob(job.id)}>
                      <strong>{job.outputFilename}</strong>
                      <span>{job.sourceType}</span>
                      <span>{new Date(job.createdAt).toLocaleString()}</span>
                    </button>
                  </li>
                ))
              ) : (
                <li className="history-empty">No jobs yet.</li>
              )}
            </ul>
          </div>

          <div className="panel">
            <h2>Job detail</h2>
            {selectedJob ? (
              <div className="stack">
                <div className="actions">
                  <button onClick={() => downloadJob(selectedJob.id, selectedJob.outputFilename)}>Download merged PDF</button>
                  <button className="ghost" onClick={() => deleteJob(selectedJob.id)}>Delete job</button>
                </div>
                <table>
                  <thead>
                    <tr>
                      <th>Order</th>
                      <th>Name</th>
                      <th>Source</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedJob.files?.map((file) => (
                      <tr key={file.id}>
                        <td>{file.order}</td>
                        <td>{file.name}</td>
                        <td>{file.sourceKind}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p>Select a job to inspect its source files and download the merged output.</p>
            )}
          </div>
        </section>
      ) : null}
    </main>
  );
}

async function api<T>(path: string, options: { method?: string; token?: string; body?: BodyInit | null } = {}): Promise<T> {
  const headers = new Headers();
  if (!(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  if (options.token) {
    headers.set("Authorization", `Bearer ${options.token}`);
  }

  const response = await fetch(`${apiBaseURL}${path}`, {
    method: options.method ?? "GET",
    headers,
    body: options.body
  });

  if (!response.ok) {
    throw new Error(await readError(response));
  }

  return response.json() as Promise<T>;
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { error?: string };
    return payload.error ?? `Request failed with status ${response.status}`;
  } catch {
    return `Request failed with status ${response.status}`;
  }
}

async function downloadResponse(response: Response, fallbackName: string) {
  const blob = await response.blob();
  const downloadURL = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const contentDisposition = response.headers.get("Content-Disposition");
  const match = contentDisposition?.match(/filename="(.+)"/);
  link.href = downloadURL;
  link.download = match?.[1] ?? fallbackName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(downloadURL);
}

function getErrorMessage(error: unknown) {
  if (error instanceof Error) {
    return error.message;
  }
  return "Unexpected error";
}

function formatBytes(bytes: number) {
  if (!bytes) {
    return "-";
  }
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(size >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

export default App;
