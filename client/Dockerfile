# QuantMind Frontend — Multi-stage Dockerfile
#
# Stage 1 (builder): npm install + vite build → produces /app/dist
# Stage 2 (runtime): nginx serves the static /dist files
#
# WHY nginx (not node serve):
#   nginx is purpose-built for serving static files — handles gzip,
#   caching headers, and concurrent connections far better than
#   `node serve`. The final image is ~25MB vs ~150MB for node.
#
# WHY build-time ARG for VITE_API_URL:
#   Vite bakes env vars into the JS bundle at build time (not runtime).
#   So VITE_API_URL must be passed as a Docker build ARG, not a runtime ENV.
#   Pass it: docker build --build-arg VITE_API_URL=https://your-backend.onrender.com .

# ─────────────────────────────────────────────────────────────
# Stage 1: Builder — npm install + vite build
# ─────────────────────────────────────────────────────────────
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files first (layer cache: only re-run npm ci if package-lock changes)
COPY package.json package-lock.json ./
RUN npm ci --frozen-lockfile

# Copy source and build
COPY . .

# VITE_API_URL is baked into the JS bundle at build time
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build
# Output: /app/dist/

# ─────────────────────────────────────────────────────────────
# Stage 2: Runtime — nginx serves /dist
# ─────────────────────────────────────────────────────────────
FROM nginx:alpine AS runtime

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Custom nginx config:
#   - Serves React SPA (all routes → index.html for client-side routing)
#   - Proxies /api/* → FastAPI backend (same as Vite proxy in dev)
#   - Enables gzip compression
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

# nginx runs as root by default in the official image — acceptable for containers
CMD ["nginx", "-g", "daemon off;"]
