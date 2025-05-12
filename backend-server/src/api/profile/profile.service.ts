import prisma from "../../utils/db"
import { getPuuid } from "../../utils/RiotRequestManager";


// data access layer for profile
// and all logic handler

export const findUserFromDb = async (gameName: string, gameTag: string) => {
    return prisma.user.findFirst({
        where :{ gameName, gameTag }
    });
}  

const Hello = async () => {
    console.log(await findUserFromDb("teddio", "0000"));
    for (let i = 0; i < 125; i++) {
        console.log(i)
        console.log(await getPuuid("teddio", "0000"));
    }
}

Hello()