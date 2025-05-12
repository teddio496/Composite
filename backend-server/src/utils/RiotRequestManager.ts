// instantiate the singleton instance of the riot api reqeuster
import { RequestManager } from './RequestManager';
import { Account, RiotMatchRegion } from '../types/riot';

export const RiotApiRequester = new RequestManager([
    { limit: 20, intervalMs: 1000 },    // 20 per second
    { limit: 100, intervalMs: 120000 }  // 100 per 2 minutes
]);

export const getPuuid = async (
    gameName: string, 
    tagline: string, 
    region : RiotMatchRegion = "americas"
): Promise<Account> => {

    const result = await RiotApiRequester.enqueue(() => 
        fetch(`https://${region}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/${gameName}/${tagline}?api_key=${process.env.RIOT_API_KEY}`))

    return await result.data?.json()
}

export const getMatchIds = async (
    puuid: string,
    region: RiotMatchRegion = "americas",
    start: number = 0,
    count: number = 100 
): Promise<string[]> => {

    const result = await RiotApiRequester.enqueue(() => 
        fetch(`https://${region}.api.riotgames.com/riot/matches/v1/matches/by-puuid/${puuid}/ids?start=${start}&count=${count}&api_key=${process.env.RIOT_API_KEY}`))
    
    return await result.data?.json()
}
export const getMatchData = async (
    matchId: string,
    region: RiotMatchRegion = "americas"
): Promise<any> => {
    const result = await RiotApiRequester.enqueue(() =>
        fetch(`https://${region}.api.riotgames.com/riot/matches/v1/matches/${matchId}?api_key=${process.env.RIOT_API_KEY}`))
    return await result.data?.json()
}   

