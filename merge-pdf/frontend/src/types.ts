export type Role = "admin" | "user";

export type User = {
  id: number;
  email: string;
  role: Role;
};

export type DrivePreviewFile = {
  sourceId: string;
  name: string;
  size: number;
  extractedOrder: number;
  webViewLink: string;
};

export type UploadReviewFile = {
  file: File;
  order: number;
};

export type JobFile = {
  id: number;
  jobId: number;
  sourceKind: string;
  name: string;
  order: number;
  size?: number;
  driveFileId?: string;
  driveLink?: string;
};

export type Job = {
  id: number;
  userId: number;
  sourceType: "drive" | "upload";
  status: "pending" | "running" | "completed" | "failed";
  outputFilename: string;
  progressPercent: number;
  errorMessage?: string;
  createdAt: string;
  files?: JobFile[];
};
