// Per-tab session id (no cookies)
export function getSid() {
    let sid = sessionStorage.getItem("sid");
    if (!sid) {
      sid = (crypto?.randomUUID?.() || Math.random().toString(36).slice(2)) + Date.now().toString(36);
      sessionStorage.setItem("sid", sid);
    }
    return sid;
  }
  