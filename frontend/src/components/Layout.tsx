import { Link, NavLink, Outlet, useNavigate } from "react-router-dom";
import { logout } from "@/api/auth";
import { User } from "@/api/client";

export default function Layout({ user, onSignOut }: { user: User; onSignOut: () => void }) {
  const navigate = useNavigate();

  async function handleSignOut() {
    await logout();
    onSignOut();
    navigate("/login");
  }

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-2 rounded-md text-sm font-medium ${
      isActive ? "bg-brand-600 text-white" : "text-slate-700 hover:bg-slate-200"
    }`;

  return (
    <div className="min-h-full">
      <nav className="bg-white border-b border-slate-200">
        <div className="container flex items-center justify-between h-14">
          <Link to="/" className="font-bold text-brand-700 text-lg">
            Agro-Vision
          </Link>
          <div className="flex items-center gap-2">
            <NavLink to="/dashboard" className={linkClass}>
              Dashboard
            </NavLink>
            <NavLink to="/upload" className={linkClass}>
              Upload
            </NavLink>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-sm text-slate-600">{user.display_name}</span>
            <button
              onClick={handleSignOut}
              className="text-sm text-slate-600 hover:text-brand-700"
            >
              Sign out
            </button>
          </div>
        </div>
      </nav>
      <main className="container py-8">
        <Outlet />
      </main>
    </div>
  );
}
