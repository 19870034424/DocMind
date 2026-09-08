interface Props {
  message?: string;
  onRetry?: () => void;
}

export default function ErrorRetry({ message, onRetry }: Props) {
  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center",
      gap: 12, padding: 40, textAlign: "center",
    }}>
      <p style={{ color: "var(--ink-muted)", margin: 0, maxWidth: 400 }}>
        {message || "加载失败，请稍后重试"}
      </p>
      {onRetry && (
        <button className="btn btn-sm" onClick={onRetry}>
          重试
        </button>
      )}
    </div>
  );
}
