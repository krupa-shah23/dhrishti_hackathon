# AI Video Analytics Dashboard

## Tech Stack

React.js, Vite, React Router, Recharts, and Lucide React.

## Installation

```bash
npm install
```

## Run

```bash
npm run dev
```

## Build

```bash
npm run build
```

## Routes

- `/` — Overview
- `/upload` — Upload Video
- `/processing/:videoId` — Processing Video
- `/analysis/:videoId` — Analysis Report
- `/previous-records` — Previous Records

Video uploads, processing, and analysis are simulated entirely in the frontend. The `VideoContext` persists mock video records in local storage, leaving the application ready for real API integration later.
