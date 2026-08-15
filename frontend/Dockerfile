# --- Build stage: compile the Vite app to static assets ---------------------
FROM node:22-alpine AS build

WORKDIR /app

# Copy the manifests first so `npm ci` is cached independently of source edits.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# No VITE_API_BASE_URL is set here on purpose: the bundle falls back to the
# relative "/api", which keeps the API host out of the build and lets Nginx
# route it at runtime. (.dockerignore keeps the developer's .env out too.)
RUN npm run build

# --- Serve stage: static files + reverse proxy ------------------------------
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
