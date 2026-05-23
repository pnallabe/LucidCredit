# ──────────────────────────────────────────────────────────────────────────────
# LucidCredit Frontend — Production Dockerfile (Next.js standalone)
# Multi-stage: deps + build in builder, then minimal standalone runtime image.
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: install dependencies ─────────────────────────────────────────────
FROM node:20-alpine AS deps

WORKDIR /app

# Copy lockfile first for layer caching
COPY package.json package-lock.json* ./
RUN npm ci --prefer-offline

# ── Stage 2: build ────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build args that must be set at build time for Next.js NEXT_PUBLIC_ inlining
ARG NEXT_PUBLIC_API_BASE_URL=http://copilot-api:8090
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}

# Ensure public/ exists so the runtime COPY step never fails for projects with no static assets
RUN mkdir -p /app/public

RUN npm run build

# ── Stage 3: lean runtime ─────────────────────────────────────────────────────
FROM node:20-alpine AS runtime

LABEL org.opencontainers.image.source="https://github.com/swarnabale/lucidcredit" \
      org.opencontainers.image.description="LucidCredit Copilot UI" \
      org.opencontainers.image.licenses="MIT"

ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

# Non-root user
RUN addgroup --system --gid 1001 nextjs && \
    adduser  --system --uid 1001 --ingroup nextjs nextjs

WORKDIR /app

# Copy Next.js standalone bundle
COPY --from=builder --chown=nextjs:nextjs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nextjs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nextjs /app/public ./public

USER nextjs

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD wget -qO- http://localhost:3000/ || exit 1

CMD ["node", "server.js"]
