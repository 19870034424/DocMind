import { useEffect, useState, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Link } from "react-router-dom";
import { listDocuments, askQuestionStream, getChatHistory, clearChatHistory, ChatMessage, DocListItem } from "../services/api";
import ErrorRetry from "../components/ErrorRetry";
import AgentChatBox from "../components/AgentChatBox";
import "./ChatPage.css";

export default function ChatPage() {
  const [docs, setDocs] = useState<DocListItem[]>([]);
  const [activeDocId, setActiveDocId] = useState<string>("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [docLoaded, setDocLoaded] = useState(false);
  const [docError, setDocError] = useState(false);
  const [chatMode, setChatMode] = useState<'fast' | 'deep'>('fast');
  const justSwitched = useRef(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // 加载文档列表
  const loadDocs = () => {
    setDocError(false);
    setDocLoaded(false);
    listDocuments().then((d) => {
      const ready = d.filter((x) => x.status === "ready");
      setDocs(ready);
      if (ready.length > 0) {
        setActiveDocId((prev) => prev || ready[0].id);
      }
      setDocLoaded(true);
    }).catch(() => {
      setDocLoaded(true);
      setDocError(true);
    });
  };

  useEffect(() => { loadDocs(); }, []);

  // 切换文档时加载对话历史
  useEffect(() => {
    if (!activeDocId) return;
    justSwitched.current = true;
    getChatHistory(activeDocId).then((msgs) => {
      setMessages(msgs);
    }).catch(() => {
      setMessages([]);
    });
  }, [activeDocId]);

  // 自动滚到底部（切文档时除外）
  useEffect(() => {
    if (justSwitched.current) {
      // 切文档：滚到顶部
      justSwitched.current = false;
      const main = document.querySelector(".codex-main");
      if (main) main.scrollTop = 0;
    } else if (messages.length > 0) {
      // 新消息：滚到底部
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  // 发送消息
  async function handleSend() {
    const question = input.trim();
    if (!question || loading || !activeDocId) return;
    setInput("");
    setError("");
    const userMsg: ChatMessage = { role: "user", content: question };
    setMessages((prev) => [...prev, userMsg, { role: "assistant", content: "" }]);
    setLoading(true);
    try {
      await askQuestionStream(
        activeDocId,
        question,
        (text) => {
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last && last.role === "assistant") {
              updated[updated.length - 1] = { ...last, content: last.content + text };
            }
            return updated;
          });
        },
      );
    } catch (e: any) {
      setError(e.message || "发送失败");
      setMessages((prev) => prev.slice(0, -2));
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  // 开启新对话 (清空当前文档历史)
  async function handleNewChat() {
    if (!activeDocId) return;
    await clearChatHistory(activeDocId);
    setMessages([]);
    setError("");
    inputRef.current?.focus();
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const activeDoc = docs.find((d) => d.id === activeDocId);

  return (
    <div className="codex-layout">
      {/* ── 左侧边栏：对话列表 ── */}
      <aside className={`codex-sidebar ${sidebarOpen ? "" : "collapsed"}`}>
        <div className="codex-sidebar-header">
          <button className="codex-new-btn" onClick={handleNewChat} disabled={!activeDocId}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
            </svg>
            新对话
          </button>
        </div>

        <div className="codex-sidebar-docs">
          {docError ? (
            <ErrorRetry message="文档列表加载失败" onRetry={loadDocs} />
          ) : docs.length === 0 ? (
            <div className="codex-sidebar-empty">
              <p>暂无文档</p>
              <Link to="/kb" className="btn btn-sm">上传文档</Link>
            </div>
          ) : (
            docs.map((doc) => (
              <button
                key={doc.id}
                className={`codex-thread ${doc.id === activeDocId ? "active" : ""}`}
                onClick={() => setActiveDocId(doc.id)}
              >
                <span className={`codex-thread-dot ${doc.format || "md"}`} />
                <span className="codex-thread-title">{doc.title || doc.id}</span>
              </button>
            ))
          )}
        </div>

        <div className="codex-sidebar-footer">
          <Link to="/kb" className="codex-sidebar-link">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
            </svg>
            知识库
          </Link>
        </div>
      </aside>

      {/* ── 右侧：聊天主区域 ── */}
      <div className="codex-chat">
        {/* 顶栏 */}
        <div className="codex-topbar">
          <div className="codex-topbar-left">
            <button
              className="codex-toggle-sidebar"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              title="切换侧栏"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="15" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/>
              </svg>
            </button>
            {activeDoc && (
              <span className="codex-current-doc">
                <span className="codex-at">@</span>
                {activeDoc.title || activeDocId}
              </span>
            )}
          </div>
          <div className="codex-topbar-right">
            <div className="chat-mode-switch">
              <button
                className={chatMode === 'fast' ? 'active' : ''}
                onClick={() => setChatMode('fast')}
              >
                ⚡ 快速
              </button>
              <button
                className={chatMode === 'deep' ? 'active' : ''}
                onClick={() => setChatMode('deep')}
              >
                🧠 深度
              </button>
            </div>
            {chatMode === 'fast' && messages.length > 0 && (
              <span className="codex-msg-count">{messages.length} 条消息</span>
            )}
          </div>
        </div>

        {/* 对话区域 */}
        <div className="codex-main">
          {chatMode === 'deep' && activeDocId ? (
            <AgentChatBox docId={activeDocId} />
          ) : !docLoaded ? (
            <div className="codex-welcome">
              <div className="codex-cursor">加载中...</div>
            </div>
          ) : !activeDocId ? (
            <div className="codex-welcome">
              <div className="codex-logo">
                <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
                  <rect width="40" height="40" rx="10" fill="var(--ink)"/>
                  <path d="M11 14h18M11 20h18M11 26h12" stroke="#fff" strokeWidth="2.5" strokeLinecap="round"/>
                </svg>
              </div>
              <h1>有什么我可以帮你的？</h1>
              <p>{docs.length === 0 ? "还没有文档，请先上传教材" : "从左侧选择一个文档开始对话"}</p>
              {docs.length === 0 && (
                <Link to="/kb" className="btn">上传文档</Link>
              )}
            </div>
          ) : (
            <div className="codex-conversation">
              {messages.length === 0 && (
                <div className="codex-start">
                  <p className="codex-hint">向这份文档提问，AI 将从文档中检索相关内容后回答</p>
                  <div className="codex-prompts">
                    {["这篇文档的主要内容是什么？", "文档的核心观点有哪些？", "总结文档的关键数据和结论"].map((q) => (
                      <button key={q} className="codex-prompt-btn" onClick={() => { setInput(q); inputRef.current?.focus(); }}>
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((msg, i) => (
                <div key={i} className={`codex-msg ${msg.role}`}>
                  <div className="codex-msg-avatar">
                    {msg.role === "user" ? (
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
                    ) : (
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                    )}
                  </div>
                  <div className="codex-msg-inner">
                    <div className="codex-msg-content">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  </div>
                </div>
              ))}

              {loading && messages.length > 0 && messages[messages.length - 1]?.content === "" && (
                <div className="codex-msg assistant">
                  <div className="codex-msg-avatar">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                  </div>
                  <div className="codex-msg-inner">
                    <div className="codex-msg-content">
                      <span className="codex-cursor">思考中...</span>
                    </div>
                  </div>
                </div>
              )}

              {error && <div className="codex-error">{error}</div>}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        {/* 底部输入 (仅快速模式) */}
        {activeDocId && chatMode === 'fast' && (
          <div className="codex-input-area">
            <div className="codex-input-row">
              <textarea
                ref={inputRef}
                className="codex-input"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="输入问题，Enter 发送，Shift+Enter 换行..."
                disabled={loading}
                rows={1}
                maxLength={2000}
              />
              <button className="codex-send-btn" onClick={handleSend} disabled={loading || !input.trim()}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/>
                </svg>
              </button>
            </div>
            <div className="codex-input-hint">
              Enter 发送 · Shift+Enter 换行
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
