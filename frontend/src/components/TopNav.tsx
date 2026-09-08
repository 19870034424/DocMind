import { NavLink } from "react-router-dom";
import "./TopNav.css";

export default function TopNav() {
  return (
    <header className="topnav">
      <div className="topnav-inner">
        <NavLink to="/" className="topnav-brand">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
            <rect width="24" height="24" rx="6" fill="var(--ink)"/>
            <path d="M7 8h10M7 12h10M7 16h6" stroke="#fff" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          <span>DocMind</span>
        </NavLink>

        <nav className="topnav-links">
          <NavLink
            to="/"
            end
            className={({ isActive }) => `topnav-link ${isActive ? "active" : ""}`}
          >
            问答对话
          </NavLink>
          <NavLink
            to="/kb"
            className={({ isActive }) => `topnav-link ${isActive ? "active" : ""}`}
          >
            本地知识库
          </NavLink>
        </nav>
      </div>
    </header>
  );
}
