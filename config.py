HOUSE_CONFIG = {
	"MSSubClass": {
		"description": "Identifies the type of dwelling involved in the sale.",
		"dtype": "object",
		"imputation_method": "",
		"members": {
			"20": { "ordinal":0, "name":"1-STORY 1946 & NEWER ALL STYLES"},
			"30": { "ordinal":0, "name":"1-STORY 1945 & OLDER"},
			"40": { "ordinal":0, "name":"1-STORY W/FINISHED ATTIC ALL AGES"},
			"45": { "ordinal":0, "name":"1-1/2 STORY - UNFINISHED ALL AGES"},
			"50": { "ordinal":0, "name":"1-1/2 STORY FINISHED ALL AGES"},
			"60": { "ordinal":0, "name":"2-STORY 1946 & NEWER"},
			"70": { "ordinal":0, "name":"2-STORY 1945 & OLDER"},
			"75": { "ordinal":0, "name":"2-1/2 STORY ALL AGES"},
			"80": { "ordinal":0, "name":"SPLIT OR MULTI-LEVEL"},
			"85": { "ordinal":0, "name":"SPLIT FOYER"},
			"90": { "ordinal":0, "name":"DUPLEX - ALL STYLES AND AGES"},
			"120": { "ordinal":0, "name":"1-STORY PUD (Planned Unit Development) - 1946 & NEWER"},
			"150": { "ordinal":0, "name":"1-1/2 STORY PUD - ALL AGES"},
			"160": { "ordinal":0, "name":"2-STORY PUD - 1946 & NEWER"},
			"180": { "ordinal":0, "name":"PUD - MULTILEVEL - INCL SPLIT LEV/FOYER"},
			"190": { "ordinal":0, "name":"2 FAMILY CONVERSION - ALL STYLES AND AGES"},
		}
	},
	"MSZoning": {
		"description": "Identifies the general zoning classification of the sale.",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"A": { "ordinal":0, "name":"Agriculture"},
			"C": { "ordinal":0, "name":"Commercial"},
			"FV": { "ordinal":0, "name":"Floating Village Residential"},
			"I": { "ordinal":0, "name":"Industrial"},
			"RH": { "ordinal":0, "name":"Residential High Density"},
			"RL": { "ordinal":0, "name":"Residential Low Density"},
			"RP": { "ordinal":0, "name":"Residential Low Density Park "},
			"RM": { "ordinal":0, "name":"Residential Medium Density"},
		}
	},
	"LotFrontage": {
		"description": "Linear feet of street connected to property",
		"dtype": "float64",
		"imputation_method": "",
		"members": {
		}
	},
	"LotArea": {
		"description": "Lot size in square feet",
		"dtype": "float64",
		"imputation_method": "",
		"members": {
		}
	},
	"Street": {
		"description": "Type of road access to property",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Grvl": { "ordinal":0, "name":"Gravel	"},
			"Pave": { "ordinal":0, "name":"Paved"},
		}
	},
	"Alley": {
		"description": "Type of alley access to property",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Grvl": { "ordinal":1, "name":"Gravel"},
			"Pave": { "ordinal":2, "name":"Paved"},
			"None": { "ordinal":2, "name":"No alley access"},
		}
	},
	"LotShape": {
		"description": "General shape of property",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Reg": { "ordinal":0, "name":"Regular	"},
			"IR1": { "ordinal":0, "name":"Slightly irregular"},
			"IR2": { "ordinal":0, "name":"Moderately Irregular"},
			"IR3": { "ordinal":0, "name":"Irregular"},
		}
	},
	"LandContour": {
		"description": "Flatness of the property",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Lvl": { "ordinal":0, "name":"Near Flat/Level	"},
			"Bnk": { "ordinal":0, "name":"Banked - Quick and significant rise from street grade to building"},
			"HLS": { "ordinal":0, "name":"Hillside - Significant slope from side to side"},
			"Low": { "ordinal":0, "name":"Depression"},
		}
	},
	"Utilities": {
		"description": "Type of utilities available",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"AllPub": { "ordinal":0, "name":"All public Utilities (E,G,W,& S)	"},
			"NoSewr": { "ordinal":0, "name":"Electricity, Gas, and Water (Septic Tank)"},
			"NoSeWa": { "ordinal":0, "name":"Electricity and Gas Only"},
			"ELO": { "ordinal":0, "name":"Electricity only	"},
		}
	},
	"LotConfig": {
		"description": "Lot configuration",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Inside": { "ordinal":0, "name":"Inside lot"},
			"Corner": { "ordinal":0, "name":"Corner lot"},
			"CulDSac": { "ordinal":0, "name":"Cul-de-sac"},
			"FR2": { "ordinal":0, "name":"Frontage on 2 sides of property"},
			"FR3": { "ordinal":0, "name":"Frontage on 3 sides of property"},
		}
	},
	"LandSlope": {
		"description": "Slope of property",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Gtl": { "ordinal":0, "name":"Gentle slope"},
			"Mod": { "ordinal":0, "name":"Moderate Slope	"},
			"Sev": { "ordinal":0, "name":"Severe Slope"},
		}
	},
	"Neighborhood": {
		"description": "Physical locations within Ames city limits",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Blmngtn": { "ordinal":0, "name":"Bloomington Heights"},
			"Blueste": { "ordinal":0, "name":"Bluestem"},
			"BrDale": { "ordinal":0, "name":"Briardale"},
			"BrkSide": { "ordinal":0, "name":"Brookside"},
			"ClearCr": { "ordinal":0, "name":"Clear Creek"},
			"CollgCr": { "ordinal":0, "name":"College Creek"},
			"Crawfor": { "ordinal":0, "name":"Crawford"},
			"Edwards": { "ordinal":0, "name":"Edwards"},
			"Gilbert": { "ordinal":0, "name":"Gilbert"},
			"IDOTRR": { "ordinal":0, "name":"Iowa DOT and Rail Road"},
			"MeadowV": { "ordinal":0, "name":"Meadow Village"},
			"Mitchel": { "ordinal":0, "name":"Mitchell"},
			"Names": { "ordinal":0, "name":"North Ames"},
			"NoRidge": { "ordinal":0, "name":"Northridge"},
			"NPkVill": { "ordinal":0, "name":"Northpark Villa"},
			"NridgHt": { "ordinal":0, "name":"Northridge Heights"},
			"NWAmes": { "ordinal":0, "name":"Northwest Ames"},
			"OldTown": { "ordinal":0, "name":"Old Town"},
			"SWISU": { "ordinal":0, "name":"South & West of Iowa State University"},
			"Sawyer": { "ordinal":0, "name":"Sawyer"},
			"SawyerW": { "ordinal":0, "name":"Sawyer West"},
			"Somerst": { "ordinal":0, "name":"Somerset"},
			"StoneBr": { "ordinal":0, "name":"Stone Brook"},
			"Timber": { "ordinal":0, "name":"Timberland"},
			"Veenker": { "ordinal":0, "name":"Veenker"},
		}
	},
	"Condition1": {
		"description": "Proximity to various conditions",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Artery": { "ordinal":3, "name":"Adjacent to arterial street"},
			"Feedr": { "ordinal":3, "name":"Adjacent to feeder street	"},
			"Norm": { "ordinal":2, "name":"Normal	"},
			"RRNn": { "ordinal":1, "name":"Within 200' of North-South Railroad"},
			"RRAn": { "ordinal":3, "name":"Adjacent to North-South Railroad"},
			"PosN": { "ordinal":1, "name":"Near positive off-site feature--park, greenbelt, etc."},
			"PosA": { "ordinal":1, "name":"Adjacent to postive off-site feature"},
			"RRNe": { "ordinal":2, "name":"Within 200' of East-West Railroad"},
			"RRAe": { "ordinal":3, "name":"Adjacent to East-West Railroad"},
		}
	},
	"Condition2": {
		"description": "Proximity to various conditions (if more than one is present)",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Artery": { "ordinal":3, "name":"Adjacent to arterial street"},
			"Feedr": { "ordinal":3, "name":"Adjacent to feeder street	"},
			"Norm": { "ordinal":2, "name":"Normal	"},
			"RRNn": { "ordinal":3, "name":"Within 200' of North-South Railroad"},
			"RRAn": { "ordinal":3, "name":"Adjacent to North-South Railroad"},
			"PosN": { "ordinal":1, "name":"Near positive off-site feature--park, greenbelt, etc."},
			"PosA": { "ordinal":3, "name":"Adjacent to postive off-site feature"},
			"RRNe": { "ordinal":3, "name":"Within 200' of East-West Railroad"},
			"RRAe": { "ordinal":3, "name":"Adjacent to East-West Railroad"},
		}
	},
	"BldgType": {
		"description": "Type of dwelling",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"1Fam": { "ordinal":0, "name":"Single-family Detached	"},
			"2FmCon": { "ordinal":0, "name":"Two-family Conversion; originally built as one-family dwelling"},
			"Duplx": { "ordinal":0, "name":"Duplex"},
			"TwnhsE": { "ordinal":0, "name":"Townhouse End Unit"},
			"TwnhsI": { "ordinal":0, "name":"Townhouse Inside Unit"},
		}
	},
	"HouseStyle": {
		"description": "Style of dwelling",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"1Story": { "ordinal":0, "name":"One story"},
			"1.5Fin": { "ordinal":0, "name":"One and one-half story: 2nd level finished"},
			"1.5Unf": { "ordinal":0, "name":"One and one-half story: 2nd level unfinished"},
			"2Story": { "ordinal":0, "name":"Two story"},
			"2.5Fin": { "ordinal":0, "name":"Two and one-half story: 2nd level finished"},
			"2.5Unf": { "ordinal":0, "name":"Two and one-half story: 2nd level unfinished"},
			"SFoyer": { "ordinal":0, "name":"Split Foyer"},
			"SLvl": { "ordinal":0, "name":"Split Level"},
		}
	},
	"OverallQual": {
		"description": "Rates the overall material and finish of the house",
		"dtype": "object",
		"imputation_method": "",
		"members": {
			"10": { "ordinal":10, "name":"Very Excellent"},
			"9": { "ordinal":9, "name":"Excellent"},
			"8": { "ordinal":8, "name":"Very Good"},
			"7": { "ordinal":7, "name":"Good"},
			"6": { "ordinal":6, "name":"Above Average"},
			"5": { "ordinal":5, "name":"Average"},
			"4": { "ordinal":4, "name":"Below Average"},
			"3": { "ordinal":3, "name":"Fair"},
			"2": { "ordinal":2, "name":"Poor"},
			"1": { "ordinal":1, "name":"Very Poor"},
		}
	},
	"OverallCond": {
		"description": "Rates the overall condition of the house",
		"dtype": "object",
		"imputation_method": "",
		"members": {
			"10": { "ordinal":10, "name":"Very Excellent"},
			"9": { "ordinal":9, "name":"Excellent"},
			"8": { "ordinal":8, "name":"Very Good"},
			"7": { "ordinal":7, "name":"Good"},
			"6": { "ordinal":6, "name":"Above Average	"},
			"5": { "ordinal":5, "name":"Average"},
			"4": { "ordinal":4, "name":"Below Average	"},
			"3": { "ordinal":3, "name":"Fair"},
			"2": { "ordinal":2, "name":"Poor"},
			"1": { "ordinal":1, "name":"Very Poor"},
		}
	},
	"YearBuilt": {
		"description": "Original construction date",
		"dtype": "",
		"imputation_method": "",
		"members": {
		}
	},
	"YearRemodAdd": {
		"description": "Remodel date (same as construction date if no remodeling or additions)",
		"dtype": "",
		"imputation_method": "",
		"members": {
		}
	},
	"RoofStyle": {
		"description": "Type of roof",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Flat": { "ordinal":0, "name":"Flat"},
			"Gable": { "ordinal":0, "name":"Gable"},
			"Gambrel": { "ordinal":0, "name":"Gabrel (Barn)"},
			"Hip": { "ordinal":0, "name":"Hip"},
			"Mansard": { "ordinal":0, "name":"Mansard"},
			"Shed": { "ordinal":0, "name":"Shed"},
		}
	},
	"RoofMatl": {
		"description": "Roof material",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"ClyTile": { "ordinal":0, "name":"Clay or Tile"},
			"CompShg": { "ordinal":0, "name":"Standard (Composite) Shingle"},
			"Membran": { "ordinal":0, "name":"Membrane"},
			"Metal": { "ordinal":0, "name":"Metal"},
			"Roll": { "ordinal":0, "name":"Roll"},
			"Tar&Grv": { "ordinal":0, "name":"Gravel & Tar"},
			"WdShake": { "ordinal":0, "name":"Wood Shakes"},
			"WdShngl": { "ordinal":0, "name":"Wood Shingles"},
		}
	},
	"Exterior1st": {
		"description": "Exterior covering on house",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"AsbShng": { "ordinal":0, "name":"Asbestos Shingles"},
			"AsphShn": { "ordinal":0, "name":"Asphalt Shingles"},
			"BrkComm": { "ordinal":0, "name":"Brick Common"},
			"BrkFace": { "ordinal":0, "name":"Brick Face"},
			"CBlock": { "ordinal":0, "name":"Cinder Block"},
			"CemntBd": { "ordinal":0, "name":"Cement Board"},
			"HdBoard": { "ordinal":0, "name":"Hard Board"},
			"ImStucc": { "ordinal":0, "name":"Imitation Stucco"},
			"MetalSd": { "ordinal":0, "name":"Metal Siding"},
			"Other": { "ordinal":0, "name":"Other"},
			"Plywood": { "ordinal":0, "name":"Plywood"},
			"PreCast": { "ordinal":0, "name":"PreCast	"},
			"Stone": { "ordinal":0, "name":"Stone"},
			"Stucco": { "ordinal":0, "name":"Stucco"},
			"VinylSd": { "ordinal":0, "name":"Vinyl Siding"},
			"Wd Sdng": { "ordinal":0, "name":"Wood Siding"},
			"WdShing": { "ordinal":0, "name":"Wood Shingles"},
		}
	},
	"Exterior2nd": {
		"description": "Exterior covering on house (if more than one material)",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"AsbShng": { "ordinal":0, "name":"Asbestos Shingles"},
			"AsphShn": { "ordinal":0, "name":"Asphalt Shingles"},
			"BrkComm": { "ordinal":0, "name":"Brick Common"},
			"BrkFace": { "ordinal":0, "name":"Brick Face"},
			"CBlock": { "ordinal":0, "name":"Cinder Block"},
			"CemntBd": { "ordinal":0, "name":"Cement Board"},
			"HdBoard": { "ordinal":0, "name":"Hard Board"},
			"ImStucc": { "ordinal":0, "name":"Imitation Stucco"},
			"MetalSd": { "ordinal":0, "name":"Metal Siding"},
			"Other": { "ordinal":0, "name":"Other"},
			"Plywood": { "ordinal":0, "name":"Plywood"},
			"PreCast": { "ordinal":0, "name":"PreCast"},
			"Stone": { "ordinal":0, "name":"Stone"},
			"Stucco": { "ordinal":0, "name":"Stucco"},
			"VinylSd": { "ordinal":0, "name":"Vinyl Siding"},
			"Wd Sdng": { "ordinal":0, "name":"Wood Siding"},
			"WdShing": { "ordinal":0, "name":"Wood Shingles"},
		}
	},
	"MasVnrType": {
		"description": "Masonry veneer type",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"BrkCmn": { "ordinal":0, "name":"Brick Common"},
			"BrkFace": { "ordinal":0, "name":"Brick Face"},
			"CBlock": { "ordinal":0, "name":"Cinder Block"},
			"None": { "ordinal":0, "name":"None"},
			"Stone": { "ordinal":0, "name":"Stone"},
		}
	},
	"MasVnrArea": {
		"description": "Masonry veneer area in square feet",
		"dtype": "",
		"imputation_method": "0",
		"members": {
		}
	},
	"ExterQual": {
		"description": "Evaluates the quality of the material on the exterior ",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Ex": { "ordinal":5, "name":"Excellent"},
			"Gd": { "ordinal":4, "name":"Good"},
			"TA": { "ordinal":3, "name":"Average/Typical"},
			"Fa": { "ordinal":2, "name":"Fair"},
			"Po": { "ordinal":1, "name":"Poor"},
		}
	},
	"ExterCond": {
		"description": "Evaluates the present condition of the material on the exterior",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Ex": { "ordinal":0, "name":"Excellent"},
			"Gd": { "ordinal":0, "name":"Good"},
			"TA": { "ordinal":0, "name":"Average/Typical"},
			"Fa": { "ordinal":0, "name":"Fair"},
			"Po": { "ordinal":0, "name":"Poor"},
		}
	},
	"Foundation": {
		"description": "Type of foundation",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"BrkTil": { "ordinal":0, "name":"Brick & Tile"},
			"CBlock": { "ordinal":0, "name":"Cinder Block"},
			"PConc": { "ordinal":0, "name":"Poured Contrete	"},
			"Slab": { "ordinal":0, "name":"Slab"},
			"Stone": { "ordinal":0, "name":"Stone"},
			"Wood": { "ordinal":0, "name":"Wood"},
		}
	},
	"BsmtQual": {
		"description": "Evaluates the height of the basement",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Ex": { "ordinal":3, "name":"Excellent (100+ inches)	"},
			"Gd": { "ordinal":2, "name":"Good (90-99 inches)"},
			"TA": { "ordinal":1, "name":"Typical (80-89 inches)"},
			"Fa": { "ordinal":1, "name":"Fair (70-79 inches)"},
			"Po": { "ordinal":1, "name":"Poor (<70 inches"},
			"None": { "ordinal":1, "name":"No Basement"},
		}
	},
	"BsmtCond": {
		"description": "Evaluates the general condition of the basement",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Ex": { "ordinal":4, "name":"Excellent"},
			"Gd": { "ordinal":3, "name":"Good"},
			"TA": { "ordinal":2, "name":"Typical - slight dampness allowed"},
			"Fa": { "ordinal":1, "name":"Fair - dampness or some cracking or settling"},
			"Po": { "ordinal":1, "name":"Poor - Severe cracking, settling, or wetness"},
			"None": { "ordinal":1, "name":"No Basement"},
		}
	},
	"BsmtExposure": {
		"description": "Refers to walkout or garden level walls",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Gd": { "ordinal":3, "name":"Good Exposure"},
			"Av": { "ordinal":2, "name":"Average Exposure (split levels or foyers typically score average or above)	"},
			"Mn": { "ordinal":2, "name":"Mimimum Exposure"},
			"No": { "ordinal":2, "name":"No Exposure"},
			"None": { "ordinal":1, "name":"No Basement"},
		}
	},
	"BsmtFinType1": {
		"description": "Rating of basement finished area",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"GLQ": { "ordinal":2, "name":"Good Living Quarters"},
			"ALQ": { "ordinal":1, "name":"Average Living Quarters"},
			"BLQ": { "ordinal":1, "name":"Below Average Living Quarters	"},
			"Rec": { "ordinal":1, "name":"Average Rec Room"},
			"LwQ": { "ordinal":1, "name":"Low Quality"},
			"Unf": { "ordinal":1, "name":"Unfinshed"},
			"None": { "ordinal":1, "name":"No Basement"},
		}
	},
	"BsmtFinSF1": {
		"description": "Type 1 finished square feet",
		"dtype": "float64",
		"imputation_method": "0",
		"members": {
		}
	},
	"BsmtFinType2": {
		"description": "Rating of basement finished area (if multiple types)",
		"dtype": "",
		"imputation_method": "0",
		"members": {
			"GLQ": { "ordinal":4, "name":"Good Living Quarters"},
			"ALQ": { "ordinal":3, "name":"Average Living Quarters"},
			"BLQ": { "ordinal":2, "name":"Below Average Living Quarters	"},
			"Rec": { "ordinal":2, "name":"Average Rec Room"},
			"LwQ": { "ordinal":2, "name":"Low Quality"},
			"Unf": { "ordinal":2, "name":"Unfinshed"},
			"None": { "ordinal":1, "name":"No Basement"},
		}
	},
	"BsmtFinSF2": {
		"description": "Type 2 finished square feet",
		"dtype": "float64",
		"imputation_method": "0",
		"members": {
		}
	},
	"BsmtUnfSF": {
		"description": "Unfinished square feet of basement area",
		"dtype": "float64",
		"imputation_method": "0",
		"members": {
		}
	},
	"TotalBsmtSF": {
		"description": "Total square feet of basement area",
		"dtype": "float64",
		"imputation_method": "0",
		"members": {
		}
	},
	"Heating": {
		"description": "Type of heating",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Floor": { "ordinal":0, "name":"Floor Furnace"},
			"GasA": { "ordinal":0, "name":"Gas forced warm air furnace"},
			"GasW": { "ordinal":0, "name":"Gas hot water or steam heat"},
			"Grav": { "ordinal":0, "name":"Gravity furnace	"},
			"OthW": { "ordinal":0, "name":"Hot water or steam heat other than gas"},
			"Wall": { "ordinal":0, "name":"Wall furnace"},
		}
	},
	"HeatingQC": {
		"description": "Heating quality and condition",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Ex": { "ordinal":5, "name":"Excellent"},
			"Gd": { "ordinal":4, "name":"Good"},
			"TA": { "ordinal":3, "name":"Average/Typical"},
			"Fa": { "ordinal":2, "name":"Fair"},
			"Po": { "ordinal":1, "name":"Poor"},
		}
	},
	"CentralAir": {
		"description": "Central air conditioning",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"N": { "ordinal":2, "name":"No"},
			"Y": { "ordinal":1, "name":"Yes"},
		}
	},
	"Electrical": {
		"description": "Electrical system",
		"dtype": "",
		"imputation_method": "SBrkr",
		"members": {
			"SBrkr": { "ordinal":3, "name":"Standard Circuit Breakers & Romex"},
			"FuseA": { "ordinal":2, "name":"Fuse Box over 60 AMP and all Romex wiring (Average)	"},
			"FuseF": { "ordinal":2, "name":"60 AMP Fuse Box and mostly Romex wiring (Fair)"},
			"FuseP": { "ordinal":1, "name":"60 AMP Fuse Box and mostly knob & tube wiring (poor)"},
			"Mix": { "ordinal":1, "name":"Mixed"},
		}
	},
	"1stFlrSF": {
		"description": "First Floor square feet",
		"dtype": "float64",
		"imputation_method": "",
		"members": {
		}
	},
	"2ndFlrSF": {
		"description": "Second floor square feet",
		"dtype": "float64",
		"imputation_method": "",
		"members": {
		}
	},
	"LowQualFinSF": {
		"description": "Low quality finished square feet (all floors)",
		"dtype": "float64",
		"imputation_method": "",
		"members": {
		}
	},
	"GrLivArea": {
		"description": "Above grade (ground) living area square feet",
		"dtype": "float64",
		"imputation_method": "",
		"members": {
		}
	},
	"BsmtFullBath": {
		"description": "Basement full bathrooms",
		"dtype": "int64",
		"imputation_method": "0",
		"members": {
		}
	},
	"BsmtHalfBath": {
		"description": "Basement half bathrooms",
		"dtype": "int64",
		"imputation_method": "0",
		"members": {
		}
	},
	"FullBath": {
		"description": "Full bathrooms above grade",
		"dtype": "int64",
		"imputation_method": "",
		"members": {
		}
	},
	"HalfBath": {
		"description": "Half baths above grade",
		"dtype": "int64",
		"imputation_method": "",
		"members": {
		}
	},
	"BedroomAbvGr": {
		"description": "Bedrooms above grade (does NOT include basement bedrooms)",
		"dtype": "int64",
		"imputation_method": "",
		"members": {
		}
	},
	"Kitchen": {
		"description": "Kitchens above grade",
		"dtype": "",
		"imputation_method": "",
		"members": {
		}
	},
	"KitchenQual": {
		"description": "Kitchen quality",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Ex": { "ordinal":6, "name":"Excellent"},
			"Gd": { "ordinal":5, "name":"Good"},
			"TA": { "ordinal":4, "name":"Typical/Average"},
			"Fa": { "ordinal":3, "name":"Fair"},
			"Po": { "ordinal":2, "name":"Poor"},
			"None": { "ordinal":1, "name":"Typical/Average"},
		}
	},
	"TotRmsAbvGrd": {
		"description": "Total rooms above grade (does not include bathrooms)",
		"dtype": "",
		"imputation_method": "",
		"members": {
		}
	},
	"Functional": {
		"description": "Home functionality (Assume typical unless deductions are warranted)",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Typ": { "ordinal":8, "name":"Typical Functionality"},
			"None": { "ordinal":8, "name":"Typical Functionality"},
			"Min1": { "ordinal":7, "name":"Minor Deductions 1"},
			"Min2": { "ordinal":6, "name":"Minor Deductions 2"},
			"Mod": { "ordinal":5, "name":"Moderate Deductions"},
			"Maj1": { "ordinal":4, "name":"Major Deductions 1"},
			"Maj2": { "ordinal":3, "name":"Major Deductions 2"},
			"Sev": { "ordinal":2, "name":"Severely Damaged"},
			"Sal": { "ordinal":1, "name":"Salvage only"},
		}
	},
	"Fireplaces": {
		"description": "Number of fireplaces",
		"dtype": "int64",
		"imputation_method": "",
		"members": {
		}
	},
	"FireplaceQu": {
		"description": "Fireplace quality",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Ex": { "ordinal":0, "name":"Excellent - Exceptional Masonry Fireplace"},
			"Gd": { "ordinal":0, "name":"Good - Masonry Fireplace in main level"},
			"TA": { "ordinal":0, "name":"Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement"},
			"Fa": { "ordinal":0, "name":"Fair - Prefabricated Fireplace in basement"},
			"Po": { "ordinal":0, "name":"Poor - Ben Franklin Stove"},
			"NA": { "ordinal":0, "name":"No Fireplace"},
		}
	},
	"GarageType": {
		"description": "Garage location",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"2Types": { "ordinal":0, "name":"More than one type of garage"},
			"Attchd": { "ordinal":0, "name":"Attached to home"},
			"Basment": { "ordinal":0, "name":"Basement Garage"},
			"BuiltIn": { "ordinal":0, "name":"Built-In (Garage part of house - typically has room above garage)"},
			"CarPort": { "ordinal":0, "name":"Car Port"},
			"Detchd": { "ordinal":0, "name":"Detached from home"},
			"NA": { "ordinal":0, "name":"No Garage"},
		}
	},
	"GarageYrBlt": {
		"description": "Year garage was built",
		"dtype": "",
		"imputation_method": "NA",
		"members": {
		}
	},
	"GarageFinish": {
		"description": "Interior finish of the garage",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Fin": { "ordinal": 4, "name":"Finished"},
			"RFn": { "ordinal": 3, "name":"Rough Finished	"},
			"Unf": { "ordinal": 2, "name":"Unfinished"},
			"None": { "ordinal": 1, "name":"No Garage"},
		}
	},
	"GarageCars": {
		"description": "Size of garage in car capacity",
		"dtype": "int64",
		"imputation_method": "0",
		"members": {
		}
	},
	"GarageArea": {
		"description": "Size of garage in square feet",
		"dtype": "float64",
		"imputation_method": "0",
		"members": {
		}
	},
	"GarageQual": {
		"description": "Garage quality",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Ex": { "ordinal":6, "name":"Excellent"},
			"Gd": { "ordinal":5, "name":"Good"},
			"TA": { "ordinal":4, "name":"Typical/Average"},
			"Fa": { "ordinal":3, "name":"Fair"},
			"Po": { "ordinal":2, "name":"Poor"},
			"None": { "ordinal":1, "name":"No Garage"},
		}
	},
	"GarageCond": {
		"description": "Garage condition",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Ex": { "ordinal":6, "name":"Excellent"},
			"Gd": { "ordinal":5, "name":"Good"},
			"TA": { "ordinal":4, "name":"Typical/Average"},
			"Fa": { "ordinal":3, "name":"Fair"},
			"Po": { "ordinal":2, "name":"Poor"},
			"None": { "ordinal":1, "name":"No Garage"},
		}
	},
	"PavedDrive": {
		"description": "Paved driveway",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Y": { "ordinal":0, "name":"Paved "},
			"P": { "ordinal":0, "name":"Partial Pavement"},
			"N": { "ordinal":0, "name":"Dirt/Gravel"},
		}
	},
	"WoodDeckSF": {
		"description": "Wood deck area in square feet",
		"dtype": "float64",
		"imputation_method": "",
		"members": {
		}
	},
	"OpenPorchSF": {
		"description": "Open porch area in square feet",
		"dtype": "float64",
		"imputation_method": "",
		"members": {
		}
	},
	"EnclosedPorch": {
		"description": "Enclosed porch area in square feet",
		"dtype": "float64",
		"imputation_method": "",
		"members": {
		}
	},
	"3SsnPorch": {
		"description": "Three season porch area in square feet",
		"dtype": "float64",
		"imputation_method": "",
		"members": {
		}
	},
	"ScreenPorch": {
		"description": "Screen porch area in square feet",
		"dtype": "float64",
		"imputation_method": "",
		"members": {
		}
	},
	"PoolArea": {
		"description": "Pool area in square feet",
		"dtype": "float64",
		"imputation_method": "",
		"members": {
		}
	},
	"PoolQC": {
		"description": "Pool quality",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Ex": { "ordinal":5, "name":"Excellent"},
			"Gd": { "ordinal":4, "name":"Good"},
			"TA": { "ordinal":3, "name":"Average/Typical"},
			"Fa": { "ordinal":2, "name":"Fair"},
			"None": { "ordinal":0, "name":"No Pool"},
			"Na": { "ordinal":0, "name":"No Pool"},
		}
	},
	"Fence": {
		"description": "Fence quality",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"GdPrv": { "ordinal":0, "name":"Good Privacy"},
			"MnPrv": { "ordinal":0, "name":"Minimum Privacy"},
			"GdWo": { "ordinal":0, "name":"Good Wood"},
			"MnWw": { "ordinal":0, "name":"Minimum Wood/Wire"},
			"NA": { "ordinal":0, "name":"No Fence"},
		}
	},
	"MiscFeature": {
		"description": "Miscellaneous feature not covered in other categories",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Elev": { "ordinal":0, "name":"Elevator"},
			"Gar2": { "ordinal":0, "name":"2nd Garage (if not described in garage section)"},
			"Othr": { "ordinal":0, "name":"Other"},
			"Shed": { "ordinal":0, "name":"Shed (over 100 SF)"},
			"TenC": { "ordinal":0, "name":"Tennis Court"},
			"NA": { "ordinal":0, "name":"None"},
		}
	},
	"MiscVal": {
		"description": "$Value of miscellaneous feature",
		"dtype": "float64",
		"imputation_method": "",
		"members": {
		}
	},
	"MoSold": {
		"description": "Month Sold (MM)",
		"dtype": "",
		"imputation_method": "",
		"members": {
		}
	},
	"YrSold": {
		"description": "Year Sold (YYYY)",
		"dtype": "",
		"imputation_method": "",
		"members": {
		}
	},
	"SaleType": {
		"description": "Type of sale",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"WD ": { "ordinal":0, "name":"Warranty Deed - Conventional"},
			"CWD": { "ordinal":0, "name":"Warranty Deed - Cash"},
			"VWD": { "ordinal":0, "name":"Warranty Deed - VA Loan"},
			"New": { "ordinal":0, "name":"Home just constructed and sold"},
			"COD": { "ordinal":0, "name":"Court Officer Deed/Estate"},
			"Con": { "ordinal":0, "name":"Contract 15% Down payment regular terms"},
			"ConLw": { "ordinal":0, "name":"Contract Low Down payment and low interest"},
			"ConLI": { "ordinal":0, "name":"Contract Low Interest"},
			"ConLD": { "ordinal":0, "name":"Contract Low Down"},
			"Oth": { "ordinal":0, "name":"Other"},
		}
	},
	"SaleCondition": {
		"description": "Condition of sale",
		"dtype": "",
		"imputation_method": "",
		"members": {
			"Normal": { "ordinal":0, "name":"Normal Sale"},
			"Abnorml": { "ordinal":0, "name":"Abnormal Sale -  trade, foreclosure, short sale"},
			"AdjLand": { "ordinal":0, "name":"Adjoining Land Purchase"},
			"Alloca": { "ordinal":0, "name":"Allocation - two linked properties with separate deeds, typically condo with a garage unit	"},
			"Family": { "ordinal":0, "name":"Sale between family members"},
			"Partial": { "ordinal":0, "name":"Home was not completed when last assessed (associated with New Homes)"},
			}
		},
	}
