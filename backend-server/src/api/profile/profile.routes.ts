import { Router } from "express";

import { grabCachedUser } from "./profile.controller";
const router = Router();

//orchestrate all profile routes
router.get("/", grabCachedUser);

export default router;