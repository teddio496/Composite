/*
  Warnings:

  - A unique constraint covering the columns `[gameName,gameTag]` on the table `User` will be added. If there are existing duplicate values, this will fail.

*/
-- CreateIndex
CREATE UNIQUE INDEX "User_gameName_gameTag_key" ON "User"("gameName", "gameTag");
