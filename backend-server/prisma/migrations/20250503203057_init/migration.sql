-- CreateTable
CREATE TABLE "User" (
    "puuid" TEXT NOT NULL,
    "gameName" TEXT NOT NULL,
    "gameTag" TEXT NOT NULL,

    CONSTRAINT "User_pkey" PRIMARY KEY ("puuid")
);

-- CreateTable
CREATE TABLE "Match" (
    "matchId" TEXT NOT NULL,
    "tempData" JSONB NOT NULL,

    CONSTRAINT "Match_pkey" PRIMARY KEY ("matchId")
);

-- CreateTable
CREATE TABLE "_MatchToUser" (
    "A" TEXT NOT NULL,
    "B" TEXT NOT NULL,

    CONSTRAINT "_MatchToUser_AB_pkey" PRIMARY KEY ("A","B")
);

-- CreateIndex
CREATE INDEX "_MatchToUser_B_index" ON "_MatchToUser"("B");

-- AddForeignKey
ALTER TABLE "_MatchToUser" ADD CONSTRAINT "_MatchToUser_A_fkey" FOREIGN KEY ("A") REFERENCES "Match"("matchId") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "_MatchToUser" ADD CONSTRAINT "_MatchToUser_B_fkey" FOREIGN KEY ("B") REFERENCES "User"("puuid") ON DELETE CASCADE ON UPDATE CASCADE;
