import { useRef, useState } from "react";
import "./FileUpload.css";

interface Props {
  onUploaded: () => void;
}

export default function FileUpload({ onUploaded }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");

  const ALLOWED = [".pdf", ".docx", ".doc", ".md", ".markdown"];

  async function handleFile(file: File) {
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!ALLOWED.includes(ext)) {
      setError(`不支持的文件格式: ${ext}，仅支持 PDF、Word、Markdown`);
      return;
    }
    setError("");
    setUploading(true);
    try {
      const { uploadDocument } = await import("../services/api");
      await uploadDocument(file);
      onUploaded();
    } catch (e: any) {
      setError(e.message || "上传失败");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="file-upload">
      <div
        className={`drop-zone ${uploading ? "uploading" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          const file = e.dataTransfer.files[0];
          if (file) handleFile(file);
        }}
      >
        {uploading ? (
          <p>正在上传并解析文档...</p>
        ) : (
          <>
            <p className="drop-text">拖拽文档到此处，或点击选择</p>
            <p className="drop-hint">支持 PDF、Word (.docx)、Markdown (.md)</p>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.doc,.md,.markdown"
          style={{ display: "none" }}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
            e.target.value = "";
          }}
        />
      </div>
      {error && <p className="upload-error">{error}</p>}
    </div>
  );
}
