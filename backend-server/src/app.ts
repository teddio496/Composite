import express from 'express';
import dotenv from 'dotenv';
import profileRoutes from "./api/profile/profile.routes"

dotenv.config();
const app = express();
app.use(express.json());

// orchestrate all routes 
app.use("/api/profile", profileRoutes);

export default app;
