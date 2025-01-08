# Rate Limit Constants
SHORT_TERM_LIMIT = 20
LONG_TERM_LIMIT = 100
RIOT_API_KEY = "RGAPI-f2cd19a1-2f43-4031-907d-208348b6d9a3"
# Country-Region Constants
COUNTRY_TO_REGION = {
    "BR1": "americas",
    "EUN1": "europe",
    "EUW1": "europe",
    "JP1": "asia",
    "KR": "asia",
    "LA1": "americas",
    "LA2": "americas",
    "NA1": "americas",
    "OC1": "americas",
    "RU": "europe",
    "TR1": "europe",
    "PH2": "asia",
    "SG2": "asia",
    "TH2": "asia",
    "TW2": "asia",
    "VN2": "asia",
}

STATS = [
    "timeCCingOthers",
    "totalDamageDealtToChampions",
    "damageDealtToTurrets",
    "totalHeal",
    "totalDamageTaken",
    "damageSelfMitigated",

]

CHALLENGE_STATS = [
    "damagePerMinute",
    "laneMinionsFirst10Minutes",
]

CHAMP_ID_TO_NAME = {
    1: 'Annie', 
    2: 'Olaf', 
    3: 'Galio', 
    4: 'TwistedFate', 
    5: 'XinZhao', 
    6: 'Urgot', 
    7: 'Leblanc', 
    8: 'Vladimir', 
    9: 'FiddleSticks', 
    10: 'Kayle', 
    11: 'MasterYi', 
    12: 'Alistar', 
    13: 'Ryze', 
    14: 'Sion', 
    15: 'Sivir', 
    16: 'Soraka', 
    17: 'Teemo', 
    18: 'Tristana',
    19: 'Warwick', 
    20: 'Nunu',
    21: 'MissFortune', 
    22: 'Ashe', 
    23: 'Tryndamere', 
    24: 'Jax', 
    25: 'Morgana', 
    26: 'Zilean', 
    27: 'Singed', 
    28: 'Evelynn', 
    29: 'Twitch', 
    30: 'Karthus', 
    31: 'Chogath', 
    32: 'Amumu', 
    33: 'Rammus', 
    34: 'Anivia', 
    35: 'Shaco', 
    36: 'DrMundo', 
    37: 'Sona', 
    38: 'Kassadin', 
    39: 'Irelia', 
    40: 'Janna', 
    41: 'Gangplank', 
    42: 'Corki', 
    43: 'Karma', 
    44: 'Taric', 
    45: 'Veigar', 
    48: 'Trundle', 
    50: 'Swain', 
    51: 'Caitlyn', 
    53: 'Blitzcrank', 
    54: 'Malphite', 
    55: 'Katarina', 
    56: 'Nocturne', 
    57: 'Maokai', 
    58: 'Renekton', 
    59: 'JarvanIV',
    60: 'Elise', 
    61: 'Orianna', 
    62: 'MonkeyKing', 
    63: 'Brand', 
    64: 'LeeSin',
    67: 'Vayne', 
    68: 'Rumble', 
    69: 'Cassiopeia', 
    72: 'Skarner', 
    74: 'Heimerdinger', 
    75: 'Nasus', 
    76: 'Nidalee', 
    77: 'Udyr', 
    78: 'Poppy', 
    79: 'Gragas', 
    80: 'Pantheon', 
    81: 'Ezreal', 
    82: 'Mordekaiser', 
    83: 'Yorick', 
    84: 'Akali', 
    85: 'Kennen', 
    86: 'Garen', 
    89: 'Leona', 
    90: 'Malzahar', 
    91: 'Talon', 
    92: 'Riven', 
    96: 'KogMaw', 
    98: 'Shen', 
    99: 'Lux', 
    101: 'Xerath', 
    102: 'Shyvana', 
    103: 'Ahri', 
    104: 'Graves', 
    105: 'Fizz', 
    106: 'Volibear', 
    107: 'Rengar', 
    110: 'Varus', 
    111: 'Nautilus', 
    112: 'Viktor', 
    113: 'Sejuani', 
    114: 'Fiora', 
    115: 'Ziggs', 
    117: 'Lulu', 
    119: 'Draven', 
    120: 'Hecarim', 
    121: 'Khazix', 
    122: 'Darius', 
    126: 'Jayce', 
    127: 'Lissandra', 
    131: 'Diana', 
    133: 'Quinn', 
    134: 'Syndra', 
    136: 'AurelionSol', 
    141: 'Kayn', 
    142: 'Zoe', 
    143: 'Zyra', 
    145: 'Kaisa', 
    147: 'Seraphine', 
    150: 'Gnar', 
    154: 'Zac', 
    157: 'Yasuo', 
    161: 'Velkoz', 
    163: 'Taliyah', 
    164: 'Camille', 
    166: 'Akshan', 
    200: 'Belveth', 
    201: 'Braum', 
    202: 'Jhin', 
    203: 'Kindred', 
    221: 'Zeri', 
    222: 'Jinx', 
    223: 'TahmKench', 
    233: 'Briar', 
    234: 'Viego', 
    235: 'Senna', 
    236: 'Lucian', 
    238: 'Zed', 
    240: 'Kled', 
    245: 'Ekko', 
    246: 'Qiyana', 
    254: 'Vi', 
    266: 'Aatrox', 
    267: 'Nami', 
    268: 'Azir', 
    350: 'Yuumi', 
    360: 'Samira', 
    412: 'Thresh', 
    420: 'Illaoi', 
    421: 'RekSai', 
    427: 'Ivern', 
    429: 'Kalista', 
    432: 'Bard', 
    497: 'Rakan', 
    498: 'Xayah', 
    516: 'Ornn', 
    517: 'Sylas', 
    518: 'Neeko', 
    523: 'Aphelios', 
    526: 'Rell', 
    555: 'Pyke', 
    711: 'Vex', 
    777: 'Yone', 
    799: 'Ambessa', 
    875: 'Sett', 
    876: 'Lillia', 
    887: 'Gwen', 
    888: 'Renata', 
    893: 'Aurora', 
    895: 'Nilah', 
    897: 'KSante', 
    901: 'Smolder', 
    902: 'Milio', 
    910: 'Hwei', 
    950: 'Naafiri'
}

# Queue Info 
QUEUE_INFO = [
    {
        "queueId": 0,
        "map": "Custom games",
        "description": "",
        "notes": ""
    },
    {
        "queueId": 2,
        "map": "Summoner's Rift",
        "description": "5v5 Blind Pick games",
        "notes": "Deprecated in patch 7.19 in favor of queueId 430"
    },
    {
        "queueId": 4,
        "map": "Summoner's Rift",
        "description": "5v5 Ranked Solo games",
        "notes": "Deprecated in favor of queueId 420"
    },
    {
        "queueId": 6,
        "map": "Summoner's Rift",
        "description": "5v5 Ranked Premade games",
        "notes": "Game mode deprecated"
    },
    {
        "queueId": 7,
        "map": "Summoner's Rift",
        "description": "Co-op vs AI games",
        "notes": "Deprecated in favor of queueId 32 and 33"
    },
    {
        "queueId": 8,
        "map": "Twisted Treeline",
        "description": "3v3 Normal games",
        "notes": "Deprecated in patch 7.19 in favor of queueId 460"
    },
    {
        "queueId": 9,
        "map": "Twisted Treeline",
        "description": "3v3 Ranked Flex games",
        "notes": "Deprecated in patch 7.19 in favor of queueId 470"
    },
    {
        "queueId": 14,
        "map": "Summoner's Rift",
        "description": "5v5 Draft Pick games",
        "notes": "Deprecated in favor of queueId 400"
    },
    {
        "queueId": 16,
        "map": "Crystal Scar",
        "description": "5v5 Dominion Blind Pick games",
        "notes": "Game mode deprecated"
    },
    {
        "queueId": 17,
        "map": "Crystal Scar",
        "description": "5v5 Dominion Draft Pick games",
        "notes": "Game mode deprecated"
    },
    {
        "queueId": 25,
        "map": "Crystal Scar",
        "description": "Dominion Co-op vs AI games",
        "notes": "Game mode deprecated"
    },
    {
        "queueId": 31,
        "map": "Summoner's Rift",
        "description": "Co-op vs AI Intro Bot games",
        "notes": "Deprecated in patch 7.19 in favor of queueId 830"
    },
    {
        "queueId": 32,
        "map": "Summoner's Rift",
        "description": "Co-op vs AI Beginner Bot games",
        "notes": "Deprecated in patch 7.19 in favor of queueId 840"
    },
    {
        "queueId": 33,
        "map": "Summoner's Rift",
        "description": "Co-op vs AI Intermediate Bot games",
        "notes": "Deprecated in patch 7.19 in favor of queueId 850"
    },
    {
        "queueId": 41,
        "map": "Twisted Treeline",
        "description": "3v3 Ranked Team games",
        "notes": "Game mode deprecated"
    },
    {
        "queueId": 42,
        "map": "Summoner's Rift",
        "description": "5v5 Ranked Team games",
        "notes": "Game mode deprecated"
    },
    {
        "queueId": 52,
        "map": "Twisted Treeline",
        "description": "Co-op vs AI games",
        "notes": "Deprecated in patch 7.19 in favor of queueId 800"
    },
    {
        "queueId": 61,
        "map": "Summoner's Rift",
        "description": "5v5 Team Builder games",
        "notes": "Game mode deprecated"
    },
    {
        "queueId": 65,
        "map": "Howling Abyss",
        "description": "5v5 ARAM games",
        "notes": "Deprecated in patch 7.19 in favor of queueId 450"
    },
    {
        "queueId": 67,
        "map": "Howling Abyss",
        "description": "ARAM Co-op vs AI games",
        "notes": "Game mode deprecated"
    },
    {
        "queueId": 70,
        "map": "Summoner's Rift",
        "description": "One for All games",
        "notes": "Deprecated in patch 8.6 in favor of queueId 1020"
    },
    {
        "queueId": 72,
        "map": "Howling Abyss",
        "description": "1v1 Snowdown Showdown games",
        "notes": ""
    },
    {
        "queueId": 73,
        "map": "Howling Abyss",
        "description": "2v2 Snowdown Showdown games",
        "notes": ""
    },
    {
        "queueId": 75,
        "map": "Summoner's Rift",
        "description": "6v6 Hexakill games",
        "notes": ""
    },
    {
        "queueId": 76,
        "map": "Summoner's Rift",
        "description": "Ultra Rapid Fire games",
        "notes": ""
    },
    {
        "queueId": 78,
        "map": "Howling Abyss",
        "description": "One For All: Mirror Mode games",
        "notes": ""
    },
    {
        "queueId": 83,
        "map": "Summoner's Rift",
        "description": "Co-op vs AI Ultra Rapid Fire games",
        "notes": ""
    },
    {
        "queueId": 91,
        "map": "Summoner's Rift",
        "description": "Doom Bots Rank 1 games",
        "notes": "Deprecated in patch 7.19 in favor of queueId 950"
    },
    {
        "queueId": 92,
        "map": "Summoner's Rift",
        "description": "Doom Bots Rank 2 games",
        "notes": "Deprecated in patch 7.19 in favor of queueId 950"
    },
    {
        "queueId": 93,
        "map": "Summoner's Rift",
        "description": "Doom Bots Rank 5 games",
        "notes": "Deprecated in patch 7.19 in favor of queueId 950"
    },
    {
        "queueId": 96,
        "map": "Crystal Scar",
        "description": "Ascension games",
        "notes": "Deprecated in patch 7.19 in favor of queueId 910"
    },
    {
        "queueId": 98,
        "map": "Twisted Treeline",
        "description": "6v6 Hexakill games",
        "notes": ""
    },
    {
        "queueId": 100,
        "map": "Butcher's Bridge",
        "description": "5v5 ARAM games",
        "notes": ""
    },
    {
        "queueId": 300,
        "map": "Howling Abyss",
        "description": "Legend of the Poro King games",
        "notes": "Deprecated in patch 7.19 in favor of queueId 920"
    },
    {
        "queueId": 310,
        "map": "Summoner's Rift",
        "description": "Nemesis games",
        "notes": ""
    },
    {
        "queueId": 313,
        "map": "Summoner's Rift",
        "description": "Black Market Brawlers games",
        "notes": ""
    },
    {
        "queueId": 315,
        "map": "Summoner's Rift",
        "description": "Nexus Siege games",
        "notes": "Deprecated in patch 7.19 in favor of queueId 940"
    },
    {
        "queueId": 317,
        "map": "Crystal Scar",
        "description": "Definitely Not Dominion games",
        "notes": ""
    },
    {
        "queueId": 318,
        "map": "Summoner's Rift",
        "description": "ARURF games",
        "notes": "Deprecated in patch 7.19 in favor of queueId 900"
    },
    {
        "queueId": 325,
        "map": "Summoner's Rift",
        "description": "All Random games",
        "notes": ""
    },
    {
        "queueId": 400,
        "map": "Summoner's Rift",
        "description": "5v5 Draft Pick games",
        "notes": ""
    },
    {
        "queueId": 410,
        "map": "Summoner's Rift",
        "description": "5v5 Ranked Dynamic games",
        "notes": "Game mode deprecated in patch 6.22"
    },
    {
        "queueId": 420,
        "map": "Summoner's Rift",
        "description": "5v5 Ranked Solo games",
        "notes": ""
    },
    {
        "queueId": 430,
        "map": "Summoner's Rift",
        "description": "5v5 Blind Pick games",
        "notes": ""
    },
    {
        "queueId": 440,
        "map": "Summoner's Rift",
        "description": "5v5 Ranked Flex games",
        "notes": ""
    },
    {
        "queueId": 450,
        "map": "Howling Abyss",
        "description": "5v5 ARAM games",
        "notes": ""
    },
    {
        "queueId": 460,
        "map": "Twisted Treeline",
        "description": "3v3 Blind Pick games",
        "notes": "Deprecated in patch 9.23"
    },
    {
        "queueId": 470,
        "map": "Twisted Treeline",
        "description": "3v3 Ranked Flex games",
        "notes": "Deprecated in patch 9.23"
    },
    {
        "queueId": 490,
        "map": "Summoner's Rift",
        "description": "Normal (Quickplay)",
        "notes": ""
    },
    {
        "queueId": 600,
        "map": "Summoner's Rift",
        "description": "Blood Hunt Assassin games",
        "notes": ""
    },
    {
        "queueId": 610,
        "map": "Cosmic Ruins",
        "description": "Dark Star: Singularity games",
        "notes": ""
    },
    {
        "queueId": 700,
        "map": "Summoner's Rift",
        "description": "Summoner's Rift Clash games",
        "notes": ""
    },
    {
        "queueId": 720,
        "map": "Howling Abyss",
        "description": "ARAM Clash games",
        "notes": ""
    },
    {
        "queueId": 800,
        "map": "Twisted Treeline",
        "description": "Co-op vs. AI Intermediate Bot games",
        "notes": "Deprecated in patch 9.23"
    },
    {
        "queueId": 810,
        "map": "Twisted Treeline",
        "description": "Co-op vs. AI Intro Bot games",
        "notes": "Deprecated in patch 9.23"
    },
    {
        "queueId": 820,
        "map": "Twisted Treeline",
        "description": "Co-op vs. AI Beginner Bot games",
        "notes": ""
    },
    {
        "queueId": 830,
        "map": "Summoner's Rift",
        "description": "Co-op vs. AI Intro Bot games",
        "notes": "Deprecated in March 2024 in favor of queueId 870"
    },
    {
        "queueId": 840,
        "map": "Summoner's Rift",
        "description": "Co-op vs. AI Beginner Bot games",
        "notes": "Deprecated in March 2024 in favor of queueId 880"
    },
    {
        "queueId": 850,
        "map": "Summoner's Rift",
        "description": "Co-op vs. AI Intermediate Bot games",
        "notes": "Deprecated in March 2024 in favor of queueId 890"
    },
    {
        "queueId": 870,
        "map": "Summoner's Rift",
        "description": "Co-op vs. AI Intro Bot games",
        "notes": ""
    },
    {
        "queueId": 880,
        "map": "Summoner's Rift",
        "description": "Co-op vs. AI Beginner Bot games",
        "notes": ""
    },
    {
        "queueId": 890,
        "map": "Summoner's Rift",
        "description": "Co-op vs. AI Intermediate Bot games",
        "notes": ""
    },
    {
        "queueId": 900,
        "map": "Summoner's Rift",
        "description": "ARURF games",
        "notes": ""
    },
    {
        "queueId": 910,
        "map": "Crystal Scar",
        "description": "Ascension games",
        "notes": ""
    },
    {
        "queueId": 920,
        "map": "Howling Abyss",
        "description": "Legend of the Poro King games",
        "notes": ""
    },
    {
        "queueId": 940,
        "map": "Summoner's Rift",
        "description": "Nexus Siege games",
        "notes": ""
    },
    {
        "queueId": 950,
        "map": "Summoner's Rift",
        "description": "Doom Bots Voting games",
        "notes": ""
    },
    {
        "queueId": 960,
        "map": "Summoner's Rift",
        "description": "Doom Bots Standard games",
        "notes": ""
    },
    {
        "queueId": 980,
        "map": "Valoran City Park",
        "description": "Star Guardian Invasion: Normal games",
        "notes": ""
    },
    {
        "queueId": 990,
        "map": "Valoran City Park",
        "description": "Star Guardian Invasion: Onslaught games",
        "notes": ""
    },
    {
        "queueId": 1000,
        "map": "Overcharge",
        "description": "PROJECT: Hunters games",
        "notes": ""
    },
    {
        "queueId": 1010,
        "map": "Summoner's Rift",
        "description": "Snow ARURF games",
        "notes": ""
    },
    {
        "queueId": 1020,
        "map": "Summoner's Rift",
        "description": "One for All games",
        "notes": ""
    },
    {
        "queueId": 1030,
        "map": "Crash Site",
        "description": "Odyssey Extraction: Intro games",
        "notes": ""
    },
    {
        "queueId": 1040,
        "map": "Crash Site",
        "description": "Odyssey Extraction: Cadet games",
        "notes": ""
    },
    {
        "queueId": 1050,
        "map": "Crash Site",
        "description": "Odyssey Extraction: Crewmember games",
        "notes": ""
    },
    {
        "queueId": 1060,
        "map": "Crash Site",
        "description": "Odyssey Extraction: Captain games",
        "notes": ""
    },
    {
        "queueId": 1070,
        "map": "Crash Site",
        "description": "Odyssey Extraction: Onslaught games",
        "notes": ""
    },
    {
        "queueId": 1090,
        "map": "Convergence",
        "description": "Teamfight Tactics games",
        "notes": ""
    },
    {
        "queueId": 1100,
        "map": "Convergence",
        "description": "Ranked Teamfight Tactics games",
        "notes": ""
    },
    {
        "queueId": 1110,
        "map": "Convergence",
        "description": "Teamfight Tactics Tutorial games",
        "notes": ""
    },
    {
        "queueId": 1111,
        "map": "Convergence",
        "description": "Teamfight Tactics test games",
        "notes": ""
    },
    {
        "queueId": 1200,
        "map": "Nexus Blitz",
        "description": "Nexus Blitz games",
        "notes": "Deprecated in patch 9.2"
    },
    {
        "queueId": 1210,
        "map": "Convergence",
        "description": "Teamfight Tactics Choncc's Treasure Mode",
        "notes": ""
    },
    {
        "queueId": 1300,
        "map": "Nexus Blitz",
        "description": "Nexus Blitz games",
        "notes": ""
    },
    {
        "queueId": 1400,
        "map": "Summoner's Rift",
        "description": "Ultimate Spellbook games",
        "notes": ""
    },
    {
        "queueId": 1700,
        "map": "Rings of Wrath",
        "description": "Arena",
        "notes": ""
    },
    {
        "queueId": 1710,
        "map": "Rings of Wrath",
        "description": "Arena",
        "notes": "16 player lobby"
    },
    {
        "queueId": 1810,
        "map": "Swarm",
        "description": "Swarm Mode Games",
        "notes": "Swarm Mode 1 player"
     },
     {
        "queueId": 1820,
        "map": "Swarm Mode Games",
        "description": "Swarm",
        "notes": "Swarm Mode 2 players"
     },
     {
        "queueId": 1830,
        "map": "Swarm Mode Games",
        "description": "Swarm",
        "notes": "Swarm Mode 3 players"
     },
     {
        "queueId": 1840,
        "map": "Swarm Mode Games",
        "description": "Swarm",
        "notes": "Swarm Mode 4 players"
     },
    {
        "queueId": 1900,
        "map": "Summoner's Rift",
        "description": "Pick URF games",
        "notes": ""
    },
    {
        "queueId": 2000,
        "map": "Summoner's Rift",
        "description": "Tutorial 1",
        "notes": ""
    },
    {
        "queueId": 2010,
        "map": "Summoner's Rift",
        "description": "Tutorial 2",
        "notes": ""
    },
    {
        "queueId": 2020,
        "map": "Summoner's Rift",
        "description": "Tutorial 3",
        "notes": ""
    }
]

