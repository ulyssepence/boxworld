# =============================================================================
# Boxworld
#
#   docker build -t boxworld .
#   docker run -p 8000:8000 -v $(pwd)/data:/app/data boxworld
# =============================================================================

# --- Frontend build ---
FROM oven/bun:1 AS frontend

WORKDIR /app/visualize
COPY visualize/package.json visualize/bun.lock ./
RUN bun install
COPY visualize/src/ src/
COPY visualize/static/ static/
RUN bun run build

# --- Runtime ---
FROM oven/bun:1-slim

WORKDIR /app
COPY --from=frontend /app/visualize/static/ visualize/static/
COPY --from=frontend /app/visualize/node_modules/ visualize/node_modules/
COPY visualize/src/ visualize/src/
COPY visualize/package.json visualize/tsconfig.json visualize/
COPY data/levels/ data/levels/

EXPOSE 8000
CMD ["bun", "run", "--cwd", "visualize", "start"]
