FROM node:20-alpine

WORKDIR /app

# Copy the manifests first so `npm ci` is cached independently of source edits.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 3000

# Dev server rather than a static build behind nginx: this is a demo vehicle
# for the pricing engine, and the API base URL is hardcoded to localhost:8000,
# so the app is expected to run alongside a developer's backend.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
