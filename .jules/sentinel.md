## 2024-05-19 - Timing Attack in Authentication and FastAPI Rate Limit flaw

**Vulnerability:**
1. The authentication check `auth.py`'s `login` function returned early if a user was not found, instead of calculating the password hash (which is computationally expensive, especially with bcrypt). This allowed attackers to enumerate existing emails using timing attacks (measuring response time).
2. The brute force rate limit check parsed the IP address directly using `request.client.host`, which could be spoofed or mask many users behind a single reverse proxy (a DoS risk where blocking the proxy blocks all users).

**Learning:**
In frameworks like FastAPI, `request.client.host` may just be the proxy's IP. Furthermore, password hashing functions take a predictable amount of time, and branching logic around them can easily lead to timing attacks.

**Prevention:**
Always compute a dummy password hash (with the same complexity) if the user isn't found. For rate limiting by IP, prefer to respect `X-Forwarded-For` or use a dedicated rate limiting middleware/service that handles proxy topologies safely.
