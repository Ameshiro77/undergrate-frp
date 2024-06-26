# 保存生成的labels。


# (解析coco的id)
valid_obj_ids = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
)

# {解析id：原始id}
original_labels_dict = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    17: 15,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
    23: 21,
    24: 22,
    25: 23,
    27: 24,
    28: 25,
    31: 26,
    32: 27,
    33: 28,
    34: 29,
    35: 30,
    36: 31,
    37: 32,
    38: 33,
    39: 34,
    40: 35,
    41: 36,
    42: 37,
    43: 38,
    44: 39,
    46: 40,
    47: 41,
    48: 42,
    49: 43,
    50: 44,
    51: 45,
    52: 46,
    53: 47,
    54: 48,
    55: 49,
    56: 50,
    57: 51,
    58: 52,
    59: 53,
    60: 54,
    61: 55,
    62: 56,
    63: 57,
    64: 58,
    65: 59,
    67: 60,
    70: 61,
    72: 62,
    73: 63,
    74: 64,
    75: 65,
    76: 66,
    77: 67,
    78: 68,
    79: 69,
    80: 70,
    81: 71,
    82: 72,
    84: 73,
    85: 74,
    86: 75,
    87: 76,
    88: 77,
    89: 78,
    90: 79,
}

# (动词id,1~117)
valid_verb_ids = list(range(1, 118))

# verb to id
verb_to_id_dict = {
    "adjust": 1,
    "assemble": 2,
    "block": 3,
    "blow": 4,
    "board": 5,
    "break": 6,
    "brush_with": 7,
    "buy": 8,
    "carry": 9,
    "catch": 10,
    "chase": 11,
    "check": 12,
    "clean": 13,
    "control": 14,
    "cook": 15,
    "cut": 16,
    "cut_with": 17,
    "direct": 18,
    "drag": 19,
    "dribble": 20,
    "drink_with": 21,
    "drive": 22,
    "dry": 23,
    "eat": 24,
    "eat_at": 25,
    "exit": 26,
    "feed": 27,
    "fill": 28,
    "flip": 29,
    "flush": 30,
    "fly": 31,
    "greet": 32,
    "grind": 33,
    "groom": 34,
    "herd": 35,
    "hit": 36,
    "hold": 37,
    "hop_on": 38,
    "hose": 39,
    "hug": 40,
    "hunt": 41,
    "inspect": 42,
    "install": 43,
    "jump": 44,
    "kick": 45,
    "kiss": 46,
    "lasso": 47,
    "launch": 48,
    "lick": 49,
    "lie_on": 50,
    "lift": 51,
    "light": 52,
    "load": 53,
    "lose": 54,
    "make": 55,
    "milk": 56,
    "move": 57,
    "no_interaction": 58,
    "open": 59,
    "operate": 60,
    "pack": 61,
    "paint": 62,
    "park": 63,
    "pay": 64,
    "peel": 65,
    "pet": 66,
    "pick": 67,
    "pick_up": 68,
    "point": 69,
    "pour": 70,
    "pull": 71,
    "push": 72,
    "race": 73,
    "read": 74,
    "release": 75,
    "repair": 76,
    "ride": 77,
    "row": 78,
    "run": 79,
    "sail": 80,
    "scratch": 81,
    "serve": 82,
    "set": 83,
    "shear": 84,
    "sign": 85,
    "sip": 86,
    "sit_at": 87,
    "sit_on": 88,
    "slide": 89,
    "smell": 90,
    "spin": 91,
    "squeeze": 92,
    "stab": 93,
    "stand_on": 94,
    "stand_under": 95,
    "stick": 96,
    "stir": 97,
    "stop_at": 98,
    "straddle": 99,
    "swing": 100,
    "tag": 101,
    "talk_on": 102,
    "teach": 103,
    "text_on": 104,
    "throw": 105,
    "tie": 106,
    "toast": 107,
    "train": 108,
    "turn": 109,
    "type_on": 110,
    "walk": 111,
    "wash": 112,
    "watch": 113,
    "wave": 114,
    "wear": 115,
    "wield": 116,
    "zip": 117,
}

# obj to id (解析的obj序号)
obj_to_id_dict = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "airplane": 5,
    "bus": 6,
    "train": 7,
    "truck": 8,
    "boat": 9,
    "traffic_light": 10,
    "fire_hydrant": 11,
    "stop_sign": 13,
    "parking_meter": 14,
    "bench": 15,
    "bird": 16,
    "cat": 17,
    "dog": 18,
    "horse": 19,
    "sheep": 20,
    "cow": 21,
    "elephant": 22,
    "bear": 23,
    "zebra": 24,
    "giraffe": 25,
    "backpack": 27,
    "umbrella": 28,
    "handbag": 31,
    "tie": 32,
    "suitcase": 33,
    "frisbee": 34,
    "skis": 35,
    "snowboard": 36,
    "sports_ball": 37,
    "kite": 38,
    "baseball_bat": 39,
    "baseball_glove": 40,
    "skateboard": 41,
    "surfboard": 42,
    "tennis_racket": 43,
    "bottle": 44,
    "wine_glass": 46,
    "cup": 47,
    "fork": 48,
    "knife": 49,
    "spoon": 50,
    "bowl": 51,
    "banana": 52,
    "apple": 53,
    "sandwich": 54,
    "orange": 55,
    "broccoli": 56,
    "carrot": 57,
    "hot_dog": 58,
    "pizza": 59,
    "donut": 60,
    "cake": 61,
    "chair": 62,
    "couch": 63,
    "potted_plant": 64,
    "bed": 65,
    "dining_table": 67,
    "toilet": 70,
    "tv": 72,
    "laptop": 73,
    "mouse": 74,
    "remote": 75,
    "keyboard": 76,
    "cell_phone": 77,
    "microwave": 78,
    "oven": 79,
    "toaster": 80,
    "sink": 81,
    "refrigerator": 82,
    "book": 84,
    "clock": 85,
    "vase": 86,
    "scissors": 87,
    "teddy_bear": 88,
    "hair_drier": 89,
    "toothbrush": 90,
}


# hoi(verb,obj) to id (序号代替)
hoi_to_id_dict = {
    (5, 5): 1,
    (18, 5): 2,
    (26, 5): 3,
    (31, 5): 4,
    (42, 5): 5,
    (53, 5): 6,
    (77, 5): 7,
    (88, 5): 8,
    (112, 5): 9,
    (58, 5): 10,
    (9, 2): 11,
    (37, 2): 12,
    (42, 2): 13,
    (44, 2): 14,
    (38, 2): 15,
    (63, 2): 16,
    (72, 2): 17,
    (76, 2): 18,
    (77, 2): 19,
    (88, 2): 20,
    (99, 2): 21,
    (111, 2): 22,
    (112, 2): 23,
    (58, 2): 24,
    (11, 16): 25,
    (27, 16): 26,
    (37, 16): 27,
    (66, 16): 28,
    (75, 16): 29,
    (113, 16): 30,
    (58, 16): 31,
    (5, 9): 32,
    (22, 9): 33,
    (26, 9): 34,
    (42, 9): 35,
    (44, 9): 36,
    (48, 9): 37,
    (76, 9): 38,
    (77, 9): 39,
    (78, 9): 40,
    (80, 9): 41,
    (88, 9): 42,
    (94, 9): 43,
    (106, 9): 44,
    (112, 9): 45,
    (58, 9): 46,
    (9, 44): 47,
    (21, 44): 48,
    (37, 44): 49,
    (42, 44): 50,
    (49, 44): 51,
    (59, 44): 52,
    (70, 44): 53,
    (58, 44): 54,
    (5, 6): 55,
    (18, 6): 56,
    (22, 6): 57,
    (26, 6): 58,
    (42, 6): 59,
    (53, 6): 60,
    (77, 6): 61,
    (88, 6): 62,
    (112, 6): 63,
    (114, 6): 64,
    (58, 6): 65,
    (5, 3): 66,
    (18, 3): 67,
    (22, 3): 68,
    (39, 3): 69,
    (42, 3): 70,
    (44, 3): 71,
    (53, 3): 72,
    (63, 3): 73,
    (77, 3): 74,
    (112, 3): 75,
    (58, 3): 76,
    (23, 17): 77,
    (27, 17): 78,
    (37, 17): 79,
    (40, 17): 80,
    (46, 17): 81,
    (66, 17): 82,
    (81, 17): 83,
    (112, 17): 84,
    (11, 17): 85,
    (58, 17): 86,
    (9, 62): 87,
    (37, 62): 88,
    (50, 62): 89,
    (88, 62): 90,
    (94, 62): 91,
    (58, 62): 92,
    (9, 63): 93,
    (50, 63): 94,
    (88, 63): 95,
    (58, 63): 96,
    (27, 21): 97,
    (35, 21): 98,
    (37, 21): 99,
    (40, 21): 100,
    (46, 21): 101,
    (47, 21): 102,
    (56, 21): 103,
    (66, 21): 104,
    (77, 21): 105,
    (111, 21): 106,
    (58, 21): 107,
    (13, 67): 108,
    (25, 67): 109,
    (87, 67): 110,
    (58, 67): 111,
    (9, 18): 112,
    (23, 18): 113,
    (27, 18): 114,
    (34, 18): 115,
    (37, 18): 116,
    (39, 18): 117,
    (40, 18): 118,
    (42, 18): 119,
    (46, 18): 120,
    (66, 18): 121,
    (79, 18): 122,
    (81, 18): 123,
    (99, 18): 124,
    (108, 18): 125,
    (111, 18): 126,
    (112, 18): 127,
    (11, 18): 128,
    (58, 18): 129,
    (27, 19): 130,
    (34, 19): 131,
    (37, 19): 132,
    (40, 19): 133,
    (44, 19): 134,
    (46, 19): 135,
    (53, 19): 136,
    (38, 19): 137,
    (66, 19): 138,
    (73, 19): 139,
    (77, 19): 140,
    (79, 19): 141,
    (99, 19): 142,
    (108, 19): 143,
    (111, 19): 144,
    (112, 19): 145,
    (58, 19): 146,
    (37, 4): 147,
    (42, 4): 148,
    (44, 4): 149,
    (38, 4): 150,
    (63, 4): 151,
    (72, 4): 152,
    (73, 4): 153,
    (77, 4): 154,
    (88, 4): 155,
    (99, 4): 156,
    (109, 4): 157,
    (111, 4): 158,
    (112, 4): 159,
    (58, 4): 160,
    (9, 1): 161,
    (32, 1): 162,
    (37, 1): 163,
    (40, 1): 164,
    (46, 1): 165,
    (93, 1): 166,
    (101, 1): 167,
    (103, 1): 168,
    (49, 1): 169,
    (58, 1): 170,
    (9, 64): 171,
    (37, 64): 172,
    (39, 64): 173,
    (58, 64): 174,
    (9, 20): 175,
    (27, 20): 176,
    (35, 20): 177,
    (37, 20): 178,
    (40, 20): 179,
    (46, 20): 180,
    (66, 20): 181,
    (77, 20): 182,
    (84, 20): 183,
    (111, 20): 184,
    (112, 20): 185,
    (58, 20): 186,
    (5, 7): 187,
    (22, 7): 188,
    (26, 7): 189,
    (53, 7): 190,
    (77, 7): 191,
    (88, 7): 192,
    (112, 7): 193,
    (58, 7): 194,
    (14, 72): 195,
    (76, 72): 196,
    (113, 72): 197,
    (58, 72): 198,
    (8, 53): 199,
    (16, 53): 200,
    (24, 53): 201,
    (37, 53): 202,
    (42, 53): 203,
    (65, 53): 204,
    (67, 53): 205,
    (90, 53): 206,
    (112, 53): 207,
    (58, 53): 208,
    (9, 27): 209,
    (37, 27): 210,
    (42, 27): 211,
    (59, 27): 212,
    (115, 27): 213,
    (58, 27): 214,
    (8, 52): 215,
    (9, 52): 216,
    (16, 52): 217,
    (24, 52): 218,
    (37, 52): 219,
    (42, 52): 220,
    (65, 52): 221,
    (67, 52): 222,
    (90, 52): 223,
    (58, 52): 224,
    (6, 39): 225,
    (9, 39): 226,
    (37, 39): 227,
    (85, 39): 228,
    (100, 39): 229,
    (105, 39): 230,
    (116, 39): 231,
    (58, 39): 232,
    (37, 40): 233,
    (115, 40): 234,
    (58, 40): 235,
    (27, 23): 236,
    (41, 23): 237,
    (113, 23): 238,
    (58, 23): 239,
    (13, 65): 240,
    (50, 65): 241,
    (88, 65): 242,
    (58, 65): 243,
    (42, 15): 244,
    (50, 15): 245,
    (88, 15): 246,
    (58, 15): 247,
    (9, 84): 248,
    (37, 84): 249,
    (59, 84): 250,
    (74, 84): 251,
    (58, 84): 252,
    (37, 51): 253,
    (97, 51): 254,
    (112, 51): 255,
    (49, 51): 256,
    (58, 51): 257,
    (16, 56): 258,
    (24, 56): 259,
    (37, 56): 260,
    (90, 56): 261,
    (97, 56): 262,
    (112, 56): 263,
    (58, 56): 264,
    (4, 61): 265,
    (9, 61): 266,
    (16, 61): 267,
    (24, 61): 268,
    (37, 61): 269,
    (52, 61): 270,
    (55, 61): 271,
    (68, 61): 272,
    (58, 61): 273,
    (9, 57): 274,
    (15, 57): 275,
    (16, 57): 276,
    (24, 57): 277,
    (37, 57): 278,
    (65, 57): 279,
    (90, 57): 280,
    (97, 57): 281,
    (112, 57): 282,
    (58, 57): 283,
    (9, 77): 284,
    (37, 77): 285,
    (74, 77): 286,
    (76, 77): 287,
    (102, 77): 288,
    (104, 77): 289,
    (58, 77): 290,
    (12, 85): 291,
    (37, 85): 292,
    (76, 85): 293,
    (83, 85): 294,
    (58, 85): 295,
    (9, 47): 296,
    (21, 47): 297,
    (37, 47): 298,
    (42, 47): 299,
    (70, 47): 300,
    (86, 47): 301,
    (90, 47): 302,
    (28, 47): 303,
    (112, 47): 304,
    (58, 47): 305,
    (8, 60): 306,
    (9, 60): 307,
    (24, 60): 308,
    (37, 60): 309,
    (55, 60): 310,
    (68, 60): 311,
    (90, 60): 312,
    (58, 60): 313,
    (27, 22): 314,
    (37, 22): 315,
    (39, 22): 316,
    (40, 22): 317,
    (46, 22): 318,
    (38, 22): 319,
    (66, 22): 320,
    (77, 22): 321,
    (111, 22): 322,
    (112, 22): 323,
    (113, 22): 324,
    (58, 22): 325,
    (40, 11): 326,
    (42, 11): 327,
    (59, 11): 328,
    (62, 11): 329,
    (58, 11): 330,
    (37, 48): 331,
    (51, 48): 332,
    (96, 48): 333,
    (49, 48): 334,
    (112, 48): 335,
    (58, 48): 336,
    (3, 34): 337,
    (10, 34): 338,
    (37, 34): 339,
    (91, 34): 340,
    (105, 34): 341,
    (58, 34): 342,
    (27, 25): 343,
    (46, 25): 344,
    (66, 25): 345,
    (77, 25): 346,
    (113, 25): 347,
    (58, 25): 348,
    (37, 89): 349,
    (60, 89): 350,
    (76, 89): 351,
    (58, 89): 352,
    (9, 31): 353,
    (37, 31): 354,
    (42, 31): 355,
    (58, 31): 356,
    (9, 58): 357,
    (15, 58): 358,
    (16, 58): 359,
    (24, 58): 360,
    (37, 58): 361,
    (55, 58): 362,
    (58, 58): 363,
    (9, 76): 364,
    (13, 76): 365,
    (37, 76): 366,
    (110, 76): 367,
    (58, 76): 368,
    (2, 38): 369,
    (9, 38): 370,
    (31, 38): 371,
    (37, 38): 372,
    (42, 38): 373,
    (48, 38): 374,
    (71, 38): 375,
    (58, 38): 376,
    (17, 49): 377,
    (37, 49): 378,
    (96, 49): 379,
    (112, 49): 380,
    (116, 49): 381,
    (49, 49): 382,
    (58, 49): 383,
    (37, 73): 384,
    (59, 73): 385,
    (74, 73): 386,
    (76, 73): 387,
    (110, 73): 388,
    (58, 73): 389,
    (13, 78): 390,
    (59, 78): 391,
    (60, 78): 392,
    (58, 78): 393,
    (14, 74): 394,
    (37, 74): 395,
    (76, 74): 396,
    (58, 74): 397,
    (8, 55): 398,
    (16, 55): 399,
    (24, 55): 400,
    (37, 55): 401,
    (42, 55): 402,
    (65, 55): 403,
    (67, 55): 404,
    (92, 55): 405,
    (112, 55): 406,
    (58, 55): 407,
    (13, 79): 408,
    (37, 79): 409,
    (42, 79): 410,
    (59, 79): 411,
    (76, 79): 412,
    (60, 79): 413,
    (58, 79): 414,
    (12, 14): 415,
    (64, 14): 416,
    (76, 14): 417,
    (58, 14): 418,
    (8, 59): 419,
    (9, 59): 420,
    (15, 59): 421,
    (16, 59): 422,
    (24, 59): 423,
    (37, 59): 424,
    (55, 59): 425,
    (68, 59): 426,
    (89, 59): 427,
    (90, 59): 428,
    (58, 59): 429,
    (13, 82): 430,
    (37, 82): 431,
    (57, 82): 432,
    (59, 82): 433,
    (58, 82): 434,
    (37, 75): 435,
    (69, 75): 436,
    (100, 75): 437,
    (58, 75): 438,
    (9, 54): 439,
    (15, 54): 440,
    (16, 54): 441,
    (24, 54): 442,
    (37, 54): 443,
    (55, 54): 444,
    (58, 54): 445,
    (17, 87): 446,
    (37, 87): 447,
    (59, 87): 448,
    (58, 87): 449,
    (13, 81): 450,
    (76, 81): 451,
    (112, 81): 452,
    (58, 81): 453,
    (9, 41): 454,
    (29, 41): 455,
    (33, 41): 456,
    (37, 41): 457,
    (44, 41): 458,
    (68, 41): 459,
    (77, 41): 460,
    (88, 41): 461,
    (94, 41): 462,
    (58, 41): 463,
    (1, 35): 464,
    (9, 35): 465,
    (37, 35): 466,
    (42, 35): 467,
    (44, 35): 468,
    (68, 35): 469,
    (76, 35): 470,
    (77, 35): 471,
    (94, 35): 472,
    (115, 35): 473,
    (58, 35): 474,
    (1, 36): 475,
    (9, 36): 476,
    (33, 36): 477,
    (37, 36): 478,
    (44, 36): 479,
    (77, 36): 480,
    (94, 36): 481,
    (115, 36): 482,
    (58, 36): 483,
    (37, 50): 484,
    (49, 50): 485,
    (112, 50): 486,
    (86, 50): 487,
    (58, 50): 488,
    (3, 37): 489,
    (9, 37): 490,
    (10, 37): 491,
    (20, 37): 492,
    (36, 37): 493,
    (37, 37): 494,
    (42, 37): 495,
    (45, 37): 496,
    (68, 37): 497,
    (82, 37): 498,
    (85, 37): 499,
    (91, 37): 500,
    (105, 37): 501,
    (58, 37): 502,
    (37, 13): 503,
    (95, 13): 504,
    (98, 13): 505,
    (58, 13): 506,
    (9, 33): 507,
    (19, 33): 508,
    (37, 33): 509,
    (40, 33): 510,
    (53, 33): 511,
    (59, 33): 512,
    (61, 33): 513,
    (68, 33): 514,
    (117, 33): 515,
    (58, 33): 516,
    (9, 42): 517,
    (19, 42): 518,
    (37, 42): 519,
    (42, 42): 520,
    (44, 42): 521,
    (50, 42): 522,
    (53, 42): 523,
    (77, 42): 524,
    (94, 42): 525,
    (88, 42): 526,
    (112, 42): 527,
    (58, 42): 528,
    (9, 88): 529,
    (37, 88): 530,
    (40, 88): 531,
    (46, 88): 532,
    (58, 88): 533,
    (9, 43): 534,
    (37, 43): 535,
    (42, 43): 536,
    (100, 43): 537,
    (58, 43): 538,
    (1, 32): 539,
    (16, 32): 540,
    (37, 32): 541,
    (42, 32): 542,
    (71, 32): 543,
    (106, 32): 544,
    (115, 32): 545,
    (58, 32): 546,
    (37, 80): 547,
    (60, 80): 548,
    (76, 80): 549,
    (58, 80): 550,
    (13, 70): 551,
    (30, 70): 552,
    (59, 70): 553,
    (76, 70): 554,
    (88, 70): 555,
    (94, 70): 556,
    (112, 70): 557,
    (58, 70): 558,
    (7, 90): 559,
    (37, 90): 560,
    (112, 90): 561,
    (58, 90): 562,
    (43, 10): 563,
    (76, 10): 564,
    (95, 10): 565,
    (98, 10): 566,
    (58, 10): 567,
    (18, 8): 568,
    (22, 8): 569,
    (42, 8): 570,
    (53, 8): 571,
    (76, 8): 572,
    (77, 8): 573,
    (88, 8): 574,
    (112, 8): 575,
    (58, 8): 576,
    (9, 28): 577,
    (37, 28): 578,
    (54, 28): 579,
    (59, 28): 580,
    (76, 28): 581,
    (83, 28): 582,
    (95, 28): 583,
    (58, 28): 584,
    (37, 86): 585,
    (55, 86): 586,
    (62, 86): 587,
    (58, 86): 588,
    (28, 46): 589,
    (37, 46): 590,
    (86, 46): 591,
    (107, 46): 592,
    (49, 46): 593,
    (112, 46): 594,
    (58, 46): 595,
    (27, 24): 596,
    (37, 24): 597,
    (66, 24): 598,
    (113, 24): 599,
    (58, 24): 600,
}

# id to obj
id_to_obj_dict = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic_light",
    11: "fire_hydrant",
    13: "stop_sign",
    14: "parking_meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports_ball",
    38: "kite",
    39: "baseball_bat",
    40: "baseball_glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis_racket",
    44: "bottle",
    46: "wine_glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot_dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted_plant",
    65: "bed",
    67: "dining_table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell_phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy_bear",
    89: "hair_drier",
    90: "toothbrush",
}

# id to verb
id_to_verb_dict = {
    1: "adjust",
    2: "assemble",
    3: "block",
    4: "blow",
    5: "board",
    6: "break",
    7: "brush_with",
    8: "buy",
    9: "carry",
    10: "catch",
    11: "chase",
    12: "check",
    13: "clean",
    14: "control",
    15: "cook",
    16: "cut",
    17: "cut_with",
    18: "direct",
    19: "drag",
    20: "dribble",
    21: "drink_with",
    22: "drive",
    23: "dry",
    24: "eat",
    25: "eat_at",
    26: "exit",
    27: "feed",
    28: "fill",
    29: "flip",
    30: "flush",
    31: "fly",
    32: "greet",
    33: "grind",
    34: "groom",
    35: "herd",
    36: "hit",
    37: "hold",
    38: "hop_on",
    39: "hose",
    40: "hug",
    41: "hunt",
    42: "inspect",
    43: "install",
    44: "jump",
    45: "kick",
    46: "kiss",
    47: "lasso",
    48: "launch",
    49: "lick",
    50: "lie_on",
    51: "lift",
    52: "light",
    53: "load",
    54: "lose",
    55: "make",
    56: "milk",
    57: "move",
    58: "no_interaction",
    59: "open",
    60: "operate",
    61: "pack",
    62: "paint",
    63: "park",
    64: "pay",
    65: "peel",
    66: "pet",
    67: "pick",
    68: "pick_up",
    69: "point",
    70: "pour",
    71: "pull",
    72: "push",
    73: "race",
    74: "read",
    75: "release",
    76: "repair",
    77: "ride",
    78: "row",
    79: "run",
    80: "sail",
    81: "scratch",
    82: "serve",
    83: "set",
    84: "shear",
    85: "sign",
    86: "sip",
    87: "sit_at",
    88: "sit_on",
    89: "slide",
    90: "smell",
    91: "spin",
    92: "squeeze",
    93: "stab",
    94: "stand_on",
    95: "stand_under",
    96: "stick",
    97: "stir",
    98: "stop_at",
    99: "straddle",
    100: "swing",
    101: "tag",
    102: "talk_on",
    103: "teach",
    104: "text_on",
    105: "throw",
    106: "tie",
    107: "toast",
    108: "train",
    109: "turn",
    110: "type_on",
    111: "walk",
    112: "wash",
    113: "watch",
    114: "wave",
    115: "wear",
    116: "wield",
    117: "zip",
}

# id to hoi
id_to_hoi_dict = {
    1: (5, 5),
    2: (18, 5),
    3: (26, 5),
    4: (31, 5),
    5: (42, 5),
    6: (53, 5),
    7: (77, 5),
    8: (88, 5),
    9: (112, 5),
    10: (58, 5),
    11: (9, 2),
    12: (37, 2),
    13: (42, 2),
    14: (44, 2),
    15: (38, 2),
    16: (63, 2),
    17: (72, 2),
    18: (76, 2),
    19: (77, 2),
    20: (88, 2),
    21: (99, 2),
    22: (111, 2),
    23: (112, 2),
    24: (58, 2),
    25: (11, 16),
    26: (27, 16),
    27: (37, 16),
    28: (66, 16),
    29: (75, 16),
    30: (113, 16),
    31: (58, 16),
    32: (5, 9),
    33: (22, 9),
    34: (26, 9),
    35: (42, 9),
    36: (44, 9),
    37: (48, 9),
    38: (76, 9),
    39: (77, 9),
    40: (78, 9),
    41: (80, 9),
    42: (88, 9),
    43: (94, 9),
    44: (106, 9),
    45: (112, 9),
    46: (58, 9),
    47: (9, 44),
    48: (21, 44),
    49: (37, 44),
    50: (42, 44),
    51: (49, 44),
    52: (59, 44),
    53: (70, 44),
    54: (58, 44),
    55: (5, 6),
    56: (18, 6),
    57: (22, 6),
    58: (26, 6),
    59: (42, 6),
    60: (53, 6),
    61: (77, 6),
    62: (88, 6),
    63: (112, 6),
    64: (114, 6),
    65: (58, 6),
    66: (5, 3),
    67: (18, 3),
    68: (22, 3),
    69: (39, 3),
    70: (42, 3),
    71: (44, 3),
    72: (53, 3),
    73: (63, 3),
    74: (77, 3),
    75: (112, 3),
    76: (58, 3),
    77: (23, 17),
    78: (27, 17),
    79: (37, 17),
    80: (40, 17),
    81: (46, 17),
    82: (66, 17),
    83: (81, 17),
    84: (112, 17),
    85: (11, 17),
    86: (58, 17),
    87: (9, 62),
    88: (37, 62),
    89: (50, 62),
    90: (88, 62),
    91: (94, 62),
    92: (58, 62),
    93: (9, 63),
    94: (50, 63),
    95: (88, 63),
    96: (58, 63),
    97: (27, 21),
    98: (35, 21),
    99: (37, 21),
    100: (40, 21),
    101: (46, 21),
    102: (47, 21),
    103: (56, 21),
    104: (66, 21),
    105: (77, 21),
    106: (111, 21),
    107: (58, 21),
    108: (13, 67),
    109: (25, 67),
    110: (87, 67),
    111: (58, 67),
    112: (9, 18),
    113: (23, 18),
    114: (27, 18),
    115: (34, 18),
    116: (37, 18),
    117: (39, 18),
    118: (40, 18),
    119: (42, 18),
    120: (46, 18),
    121: (66, 18),
    122: (79, 18),
    123: (81, 18),
    124: (99, 18),
    125: (108, 18),
    126: (111, 18),
    127: (112, 18),
    128: (11, 18),
    129: (58, 18),
    130: (27, 19),
    131: (34, 19),
    132: (37, 19),
    133: (40, 19),
    134: (44, 19),
    135: (46, 19),
    136: (53, 19),
    137: (38, 19),
    138: (66, 19),
    139: (73, 19),
    140: (77, 19),
    141: (79, 19),
    142: (99, 19),
    143: (108, 19),
    144: (111, 19),
    145: (112, 19),
    146: (58, 19),
    147: (37, 4),
    148: (42, 4),
    149: (44, 4),
    150: (38, 4),
    151: (63, 4),
    152: (72, 4),
    153: (73, 4),
    154: (77, 4),
    155: (88, 4),
    156: (99, 4),
    157: (109, 4),
    158: (111, 4),
    159: (112, 4),
    160: (58, 4),
    161: (9, 1),
    162: (32, 1),
    163: (37, 1),
    164: (40, 1),
    165: (46, 1),
    166: (93, 1),
    167: (101, 1),
    168: (103, 1),
    169: (49, 1),
    170: (58, 1),
    171: (9, 64),
    172: (37, 64),
    173: (39, 64),
    174: (58, 64),
    175: (9, 20),
    176: (27, 20),
    177: (35, 20),
    178: (37, 20),
    179: (40, 20),
    180: (46, 20),
    181: (66, 20),
    182: (77, 20),
    183: (84, 20),
    184: (111, 20),
    185: (112, 20),
    186: (58, 20),
    187: (5, 7),
    188: (22, 7),
    189: (26, 7),
    190: (53, 7),
    191: (77, 7),
    192: (88, 7),
    193: (112, 7),
    194: (58, 7),
    195: (14, 72),
    196: (76, 72),
    197: (113, 72),
    198: (58, 72),
    199: (8, 53),
    200: (16, 53),
    201: (24, 53),
    202: (37, 53),
    203: (42, 53),
    204: (65, 53),
    205: (67, 53),
    206: (90, 53),
    207: (112, 53),
    208: (58, 53),
    209: (9, 27),
    210: (37, 27),
    211: (42, 27),
    212: (59, 27),
    213: (115, 27),
    214: (58, 27),
    215: (8, 52),
    216: (9, 52),
    217: (16, 52),
    218: (24, 52),
    219: (37, 52),
    220: (42, 52),
    221: (65, 52),
    222: (67, 52),
    223: (90, 52),
    224: (58, 52),
    225: (6, 39),
    226: (9, 39),
    227: (37, 39),
    228: (85, 39),
    229: (100, 39),
    230: (105, 39),
    231: (116, 39),
    232: (58, 39),
    233: (37, 40),
    234: (115, 40),
    235: (58, 40),
    236: (27, 23),
    237: (41, 23),
    238: (113, 23),
    239: (58, 23),
    240: (13, 65),
    241: (50, 65),
    242: (88, 65),
    243: (58, 65),
    244: (42, 15),
    245: (50, 15),
    246: (88, 15),
    247: (58, 15),
    248: (9, 84),
    249: (37, 84),
    250: (59, 84),
    251: (74, 84),
    252: (58, 84),
    253: (37, 51),
    254: (97, 51),
    255: (112, 51),
    256: (49, 51),
    257: (58, 51),
    258: (16, 56),
    259: (24, 56),
    260: (37, 56),
    261: (90, 56),
    262: (97, 56),
    263: (112, 56),
    264: (58, 56),
    265: (4, 61),
    266: (9, 61),
    267: (16, 61),
    268: (24, 61),
    269: (37, 61),
    270: (52, 61),
    271: (55, 61),
    272: (68, 61),
    273: (58, 61),
    274: (9, 57),
    275: (15, 57),
    276: (16, 57),
    277: (24, 57),
    278: (37, 57),
    279: (65, 57),
    280: (90, 57),
    281: (97, 57),
    282: (112, 57),
    283: (58, 57),
    284: (9, 77),
    285: (37, 77),
    286: (74, 77),
    287: (76, 77),
    288: (102, 77),
    289: (104, 77),
    290: (58, 77),
    291: (12, 85),
    292: (37, 85),
    293: (76, 85),
    294: (83, 85),
    295: (58, 85),
    296: (9, 47),
    297: (21, 47),
    298: (37, 47),
    299: (42, 47),
    300: (70, 47),
    301: (86, 47),
    302: (90, 47),
    303: (28, 47),
    304: (112, 47),
    305: (58, 47),
    306: (8, 60),
    307: (9, 60),
    308: (24, 60),
    309: (37, 60),
    310: (55, 60),
    311: (68, 60),
    312: (90, 60),
    313: (58, 60),
    314: (27, 22),
    315: (37, 22),
    316: (39, 22),
    317: (40, 22),
    318: (46, 22),
    319: (38, 22),
    320: (66, 22),
    321: (77, 22),
    322: (111, 22),
    323: (112, 22),
    324: (113, 22),
    325: (58, 22),
    326: (40, 11),
    327: (42, 11),
    328: (59, 11),
    329: (62, 11),
    330: (58, 11),
    331: (37, 48),
    332: (51, 48),
    333: (96, 48),
    334: (49, 48),
    335: (112, 48),
    336: (58, 48),
    337: (3, 34),
    338: (10, 34),
    339: (37, 34),
    340: (91, 34),
    341: (105, 34),
    342: (58, 34),
    343: (27, 25),
    344: (46, 25),
    345: (66, 25),
    346: (77, 25),
    347: (113, 25),
    348: (58, 25),
    349: (37, 89),
    350: (60, 89),
    351: (76, 89),
    352: (58, 89),
    353: (9, 31),
    354: (37, 31),
    355: (42, 31),
    356: (58, 31),
    357: (9, 58),
    358: (15, 58),
    359: (16, 58),
    360: (24, 58),
    361: (37, 58),
    362: (55, 58),
    363: (58, 58),
    364: (9, 76),
    365: (13, 76),
    366: (37, 76),
    367: (110, 76),
    368: (58, 76),
    369: (2, 38),
    370: (9, 38),
    371: (31, 38),
    372: (37, 38),
    373: (42, 38),
    374: (48, 38),
    375: (71, 38),
    376: (58, 38),
    377: (17, 49),
    378: (37, 49),
    379: (96, 49),
    380: (112, 49),
    381: (116, 49),
    382: (49, 49),
    383: (58, 49),
    384: (37, 73),
    385: (59, 73),
    386: (74, 73),
    387: (76, 73),
    388: (110, 73),
    389: (58, 73),
    390: (13, 78),
    391: (59, 78),
    392: (60, 78),
    393: (58, 78),
    394: (14, 74),
    395: (37, 74),
    396: (76, 74),
    397: (58, 74),
    398: (8, 55),
    399: (16, 55),
    400: (24, 55),
    401: (37, 55),
    402: (42, 55),
    403: (65, 55),
    404: (67, 55),
    405: (92, 55),
    406: (112, 55),
    407: (58, 55),
    408: (13, 79),
    409: (37, 79),
    410: (42, 79),
    411: (59, 79),
    412: (76, 79),
    413: (60, 79),
    414: (58, 79),
    415: (12, 14),
    416: (64, 14),
    417: (76, 14),
    418: (58, 14),
    419: (8, 59),
    420: (9, 59),
    421: (15, 59),
    422: (16, 59),
    423: (24, 59),
    424: (37, 59),
    425: (55, 59),
    426: (68, 59),
    427: (89, 59),
    428: (90, 59),
    429: (58, 59),
    430: (13, 82),
    431: (37, 82),
    432: (57, 82),
    433: (59, 82),
    434: (58, 82),
    435: (37, 75),
    436: (69, 75),
    437: (100, 75),
    438: (58, 75),
    439: (9, 54),
    440: (15, 54),
    441: (16, 54),
    442: (24, 54),
    443: (37, 54),
    444: (55, 54),
    445: (58, 54),
    446: (17, 87),
    447: (37, 87),
    448: (59, 87),
    449: (58, 87),
    450: (13, 81),
    451: (76, 81),
    452: (112, 81),
    453: (58, 81),
    454: (9, 41),
    455: (29, 41),
    456: (33, 41),
    457: (37, 41),
    458: (44, 41),
    459: (68, 41),
    460: (77, 41),
    461: (88, 41),
    462: (94, 41),
    463: (58, 41),
    464: (1, 35),
    465: (9, 35),
    466: (37, 35),
    467: (42, 35),
    468: (44, 35),
    469: (68, 35),
    470: (76, 35),
    471: (77, 35),
    472: (94, 35),
    473: (115, 35),
    474: (58, 35),
    475: (1, 36),
    476: (9, 36),
    477: (33, 36),
    478: (37, 36),
    479: (44, 36),
    480: (77, 36),
    481: (94, 36),
    482: (115, 36),
    483: (58, 36),
    484: (37, 50),
    485: (49, 50),
    486: (112, 50),
    487: (86, 50),
    488: (58, 50),
    489: (3, 37),
    490: (9, 37),
    491: (10, 37),
    492: (20, 37),
    493: (36, 37),
    494: (37, 37),
    495: (42, 37),
    496: (45, 37),
    497: (68, 37),
    498: (82, 37),
    499: (85, 37),
    500: (91, 37),
    501: (105, 37),
    502: (58, 37),
    503: (37, 13),
    504: (95, 13),
    505: (98, 13),
    506: (58, 13),
    507: (9, 33),
    508: (19, 33),
    509: (37, 33),
    510: (40, 33),
    511: (53, 33),
    512: (59, 33),
    513: (61, 33),
    514: (68, 33),
    515: (117, 33),
    516: (58, 33),
    517: (9, 42),
    518: (19, 42),
    519: (37, 42),
    520: (42, 42),
    521: (44, 42),
    522: (50, 42),
    523: (53, 42),
    524: (77, 42),
    525: (94, 42),
    526: (88, 42),
    527: (112, 42),
    528: (58, 42),
    529: (9, 88),
    530: (37, 88),
    531: (40, 88),
    532: (46, 88),
    533: (58, 88),
    534: (9, 43),
    535: (37, 43),
    536: (42, 43),
    537: (100, 43),
    538: (58, 43),
    539: (1, 32),
    540: (16, 32),
    541: (37, 32),
    542: (42, 32),
    543: (71, 32),
    544: (106, 32),
    545: (115, 32),
    546: (58, 32),
    547: (37, 80),
    548: (60, 80),
    549: (76, 80),
    550: (58, 80),
    551: (13, 70),
    552: (30, 70),
    553: (59, 70),
    554: (76, 70),
    555: (88, 70),
    556: (94, 70),
    557: (112, 70),
    558: (58, 70),
    559: (7, 90),
    560: (37, 90),
    561: (112, 90),
    562: (58, 90),
    563: (43, 10),
    564: (76, 10),
    565: (95, 10),
    566: (98, 10),
    567: (58, 10),
    568: (18, 8),
    569: (22, 8),
    570: (42, 8),
    571: (53, 8),
    572: (76, 8),
    573: (77, 8),
    574: (88, 8),
    575: (112, 8),
    576: (58, 8),
    577: (9, 28),
    578: (37, 28),
    579: (54, 28),
    580: (59, 28),
    581: (76, 28),
    582: (83, 28),
    583: (95, 28),
    584: (58, 28),
    585: (37, 86),
    586: (55, 86),
    587: (62, 86),
    588: (58, 86),
    589: (28, 46),
    590: (37, 46),
    591: (86, 46),
    592: (107, 46),
    593: (49, 46),
    594: (112, 46),
    595: (58, 46),
    596: (27, 24),
    597: (37, 24),
    598: (66, 24),
    599: (113, 24),
    600: (58, 24),
}
