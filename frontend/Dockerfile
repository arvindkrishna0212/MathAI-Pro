# Use the official Node.js image as the base image
FROM node:18-alpine

# Set the working directory
WORKDIR /app/frontend

# Copy package.json and package-lock.json (if available) to install dependencies
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Build the React app for production
RUN npm run build

# Serve the app with Nginx or a simple static file server
# For simplicity, we\'ll use serve, install it globally
RUN npm install -g serve

EXPOSE 3000

CMD ["serve", "-s", "build", "-l", "3000"]

