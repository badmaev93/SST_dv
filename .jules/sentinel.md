## 2024-05-19 - Timing Attack in Authentication and FastAPI Rate Limit flaw

**Vulnerability:**
1. The authentication check `auth.py`'s `login` function returned early if a user was not found, instead of calculating the password hash (which is computationally expensive, especially with bcrypt). This allowed attackers to enumerate existing emails using timing attacks (measuring response time).
2. The brute force rate limit check parsed the IP address directly using `request.client.host`, which could be spoofed or mask many users behind a single reverse proxy (a DoS risk where blocking the proxy blocks all users).

**Learning:**
In frameworks like FastAPI, `request.client.host` may just be the proxy's IP. Furthermore, password hashing functions take a predictable amount of time, and branching logic around them can easily lead to timing attacks.

**Prevention:**
Always compute a dummy password hash (with the same complexity) if the user isn't found. For rate limiting by IP, prefer to respect `X-Forwarded-For` or use a dedicated rate limiting middleware/service that handles proxy topologies safely.

## 2024-11-11 - IP Spoofing Rate Limit Bypass via X-Forwarded-For
**Vulnerability:** In `auth.py`, the login rate limiter uses `request.headers.get("X-Forwarded-For").split(",")[0].strip()` to get the client IP.
**Learning:** `X-Forwarded-For` is a list of IPs. If a proxy appends the real IP, the client can spoof the first IP in the list, effectively rotating their apparent IP and bypassing brute-force protections.
**Prevention:** Only trust the right-most IPs appended by trusted reverse proxies, or configure the web framework/server to handle trusted proxies natively (e.g., using `ProxyHeadersMiddleware` in Starlette/FastAPI).

## 2025-02-14 - JWT tokens in URL Query Parameters
**Vulnerability:** The application was configured to read JSON Web Tokens (JWT) from the URL via the `?token=` query parameter as a fallback because static image tags (`<img src="...">`) don't naturally support custom HTTP headers like `Authorization`.
**Learning:** Passing authentication tokens directly in URL query parameters violates security best practices, as URLs are frequently logged in web servers, proxies, and browser histories. This poses a significant credential exposure risk.
**Prevention:** To fetch authenticated resources without exposing tokens in the URL, use Javascript's `fetch()` API with the `Authorization` header, read the response as a Blob, and set the image element's `src` using `URL.createObjectURL()`. Backend middlewares should solely read tokens from the `Authorization: Bearer` header.
