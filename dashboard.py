"""
TUS-GAN v3 — Interactive Dashboard
====================================
Streamlit app to generate synthetic 24-hour time-use diaries
using a trained WGAN-GP Generator conditioned on demographics.

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
import glob

# ---------------------------------------------------------------------------
# Path setup — allow dynamically importing Generator based on version
# ---------------------------------------------------------------------------
import importlib.util

def get_generator_class(version_key: str):
    module_name = f"generator_{version_key}"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), version_key, "generator.py")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.Generator

# ---------------------------------------------------------------------------
# Default paths (local — no HuggingFace downloads)
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "checkpoints", "final.pt"
)
DATA_NPZ_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "v3",
    "tusgan_encode.npz",
)

# ---------------------------------------------------------------------------
# Labels & colours
# ---------------------------------------------------------------------------
ACTIVITY_LABELS = {
    1: "Employment & Related",
    2: "Production for Own Use",
    3: "Unpaid Domestic Services",
    4: "Unpaid Caregiving",
    5: "Unpaid Volunteer/Community",
    6: "Learning",
    7: "Socializing & Religious",
    8: "Culture, Leisure & Sports",
    9: "Self-care & Maintenance",
}

ACTIVITY_COLORS = {
    1: "#ff7f0e",  # orange
    2: "#8c564b",  # brown
    3: "#2ca02c",  # green
    4: "#d62728",  # red
    5: "#9467bd",  # purple
    6: "#17becf",  # cyan
    7: "#e377c2",  # pink
    8: "#bcbd22",  # olive
    9: "#1f77b4",  # blue
}

AGE_LABELS = [
    "Childhood (<15)",
    "School Students (15-17)",
    "College / Early Work (18-24)",
    "Early Career (25-34)",
    "Mid-Career (35-44)",
    "Later Working (45-59)",
    "Retirement (60+)",
]

GENDER_LABELS = ["Male", "Female", "Transgender"]
MARITAL_LABELS = ["Married", "Widow/Widower", "Divorced/Separated", "Never Married"]

EDU_LABELS = {
    1: "Not literate",
    2: "Literate (No schooling)",
    3: "Literate (NFEC)",
    4: "Literate (TLC/AEC)",
    5: "Literate (Others)",
    6: "Below Primary",
    7: "Primary",
    8: "Middle",
    10: "Secondary",
    11: "Higher Secondary",
    12: "Diploma/Graduate+",
}

ACT_LABELS = {
    11: "Self-Employed (Own Account)",
    12: "Self-Employed (Employer)",
    21: "Helper in HH Enterprise",
    31: "Regular Salaried",
    41: "Casual Labour (Public)",
    51: "Casual Labour (Other)",
    81: "Seeking Work",
    91: "Student",
    92: "Domestic Duties Only",
    93: "Domestic Duties & Free Collection",
    94: "Rentier/Pensioner",
    95: "Disabled/Unable",
    97: "Other",
}

DOW_LABELS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
SECTOR_LABELS = ["Rural", "Urban"]

# ---------------------------------------------------------------------------
# State & District name mappings (from ITUS 2019 State_District_List_TUS.pdf)
# IDs are 0-indexed to match the dataset encoding (state_id = state_code - 1)
# ---------------------------------------------------------------------------
STATE_NAMES = {
    0: "Jammu & Kashmir",
    1: "Himachal Pradesh",
    2: "Punjab",
    3: "Chandigarh",
    4: "Uttarakhand",
    5: "Haryana",
    6: "Delhi",
    7: "Rajasthan",
    8: "Uttar Pradesh",
    9: "Bihar",
    10: "Sikkim",
    11: "Arunachal Pradesh",
    12: "Nagaland",
    13: "Manipur",
    14: "Mizoram",
    15: "Tripura",
    16: "Meghalaya",
    17: "Assam",
    18: "West Bengal",
    19: "Jharkhand",
    20: "Odisha",
    21: "Chhattisgarh",
    22: "Madhya Pradesh",
    23: "Gujarat",
    24: "Daman & Diu",
    25: "Dadra & Nagar Haveli",
    26: "Maharashtra",
    27: "Andhra Pradesh",
    28: "Karnataka",
    29: "Goa",
    30: "Lakshadweep",
    31: "Kerala",
    32: "Tamil Nadu",
    33: "Puducherry",
    34: "Andaman & Nicobar Islands",
    35: "Telangana",
}
DISTRICT_NAMES_BY_STATE = {
    0: {
        1: "Kupwara",
        2: "Badgam / Leh",
        3: "Kargil",
        4: "Punch",
        5: "Rajouri",
        6: "Kathua",
        7: "Baramula",
        8: "Bandipore",
        9: "Srinagar",
        10: "Ganderbal",
        11: "Pulwama",
        12: "Shupiyan",
        13: "Anantnag",
        14: "Kulgam",
        15: "Doda",
        16: "Ramban",
        17: "Kishtwar",
        18: "Udhampur",
        19: "Reasi",
        20: "Jammu",
        21: "Samba",
    },
    1: {
        0: "Chamba",
        1: "Kangra",
        2: "Lahul & Spiti",
        3: "Kullu",
        4: "Mandi",
        5: "Hamirpur",
        6: "Una",
        7: "Bilaspur",
        8: "Solan",
        9: "Sirmaur",
        10: "Shimla",
        11: "Kinnaur",
    },
    2: {
        0: "Gurdaspur",
        1: "Kapurthala",
        2: "Jalandhar",
        3: "Hoshiarpur",
        4: "Shahid Bhagat Singh Nagar",
        5: "Fatehgarh Sahib",
        6: "Ludhiana",
        7: "Moga",
        8: "Firozpur",
        9: "Muktsar",
        10: "Faridkot",
        11: "Bhatinda",
        12: "Mansa",
        13: "Patiala",
        14: "Amritsar",
        15: "Tarn Taran",
        16: "Rupnagar",
        17: "Sahibzada Ajit Singh Nagar",
        18: "Sangrur",
        19: "Barnala",
        20: "Pathankot",
        21: "Fazilka",
    },
    3: {
        0: "Chandigarh",
    },
    4: {
        0: "Uttarkashi",
        1: "Chamoli",
        2: "Rudraprayag",
        3: "Tehri Garhwal",
        4: "Dehradun",
        5: "Garhwal",
        6: "Pithoragarh",
        7: "Bageshwar",
        8: "Almora",
        9: "Champawat",
        10: "Nainital",
        11: "Udham Singh Nagar",
        12: "Hardwar",
    },
    5: {
        0: "Panchkula",
        1: "Ambala",
        2: "Yamunanagar",
        3: "Kurukshetra",
        4: "Kaithal",
        5: "Karnal",
        6: "Panipat",
        7: "Sonipat",
        8: "Jind",
        9: "Fatehabad",
        10: "Sirsa",
        11: "Hisar",
        12: "Bhiwani",
        13: "Rohtak",
        14: "Jhajjar",
        15: "Mahendragarh",
        16: "Rewari",
        17: "Gurgaon",
        18: "Mewat",
        19: "Faridabad",
        20: "Palwal",
    },
    6: {
        0: "North West",
        1: "North",
        2: "North East",
        3: "East",
        4: "New Delhi",
        5: "Central",
        6: "West",
        7: "South West",
        8: "South",
    },
    7: {
        0: "Sri Ganganagar",
        1: "Hanumangarh",
        2: "Bikaner",
        3: "Churu",
        4: "Jhunjhunun",
        5: "Alwar",
        6: "Bharatpur",
        7: "Dhaulpur",
        8: "Karauli",
        9: "Sawai Madhopur",
        10: "Dausa",
        11: "Jaipur",
        12: "Sikar",
        13: "Nagaur",
        14: "Jodhpur",
        15: "Jaisalmer",
        16: "Barmer",
        17: "Jalor",
        18: "Sirohi",
        19: "Pali",
        20: "Ajmer",
        21: "Tonk",
        22: "Bundi",
        23: "Bhilwara",
        24: "Rajsamand",
        25: "Dungarpur",
        26: "Banswara",
        27: "Chittaurgarh",
        28: "Kota",
        29: "Baran",
        30: "Jhalawar",
        31: "Udaipur",
        32: "Pratapgarh",
    },
    8: {
        0: "Saharanpur",
        1: "Muzaffarnagar",
        2: "Bijnor",
        3: "Moradabad",
        4: "Rampur",
        5: "Jyotiba Phule Nagar",
        6: "Meerut",
        7: "Baghpat",
        8: "Ghaziabad",
        9: "Gautam Buddha Nagar",
        10: "Bulandshahr",
        11: "Aligarh",
        12: "Mahamaya Nagar",
        13: "Mathura",
        14: "Agra",
        15: "Firozabad",
        16: "Mainpuri",
        17: "Budaun",
        18: "Bareilly",
        19: "Pilibhit",
        20: "Shahjahanpur",
        21: "Kheri",
        22: "Sitapur",
        23: "Hardoi",
        24: "Unnao",
        25: "Lucknow",
        26: "Rae Bareli",
        27: "Farrukhabad",
        28: "Kannauj",
        29: "Etawah",
        30: "Auraiya",
        31: "Kanpur Dehat",
        32: "Kanpur Nagar",
        33: "Jalaun",
        34: "Jhansi",
        35: "Lalitpur",
        36: "Hamirpur",
        37: "Mahoba",
        38: "Banda",
        39: "Chitrakoot",
        40: "Fatehpur",
        41: "Pratapgarh",
        42: "Kaushambi",
        43: "Allahabad",
        44: "Bara Banki",
        45: "Faizabad",
        46: "Ambedkar Nagar",
        47: "Sultanpur",
        48: "Bahraich",
        49: "Shrawasti",
        50: "Balrampur",
        51: "Gonda",
        52: "Siddharthnagar",
        53: "Basti",
        54: "Sant Kabir Nagar",
        55: "Maharajganj",
        56: "Gorakhpur",
        57: "Kushinagar",
        58: "Deoria",
        59: "Azamgarh",
        60: "Mau",
        61: "Ballia",
        62: "Jaunpur",
        63: "Ghazipur",
        64: "Chandauli",
        65: "Varanasi",
        66: "Sant Ravidas Nagar(Bhadohi)",
        67: "Mirzapur",
        68: "Sonbhadra",
        69: "Etah",
        70: "Kanshiram Nagar",
    },
    9: {
        0: "Pashchim Champaran",
        1: "Purba Champaran",
        2: "Sheohar",
        3: "Sitamarhi",
        4: "Madhubani",
        5: "Supaul",
        6: "Araria",
        7: "Kishanganj",
        8: "Purnia",
        9: "Katihar",
        10: "Madhepura",
        11: "Saharsa",
        12: "Darbhanga",
        13: "Muzaffarpur",
        14: "Gopalganj",
        15: "Siwan",
        16: "Saran",
        17: "Vaishali",
        18: "Samastipur",
        19: "Begusarai",
        20: "Khagaria",
        21: "Bhagalpur",
        22: "Banka",
        23: "Munger",
        24: "Lakhisarai",
        25: "Sheikhpura",
        26: "Nalanda",
        27: "Patna",
        28: "Bhojpur",
        29: "Buxar",
        30: "Kaimur (Bhabua)",
        31: "Rohtas",
        32: "Aurangabad",
        33: "Gaya",
        34: "Nawada",
        35: "Jamui",
        36: "Jehanabad",
        37: "Arwal",
    },
    10: {
        0: "North District",
        1: "West District",
        2: "South District",
        3: "East District",
    },
    11: {
        0: "Tawang",
        1: "West Kameng",
        2: "East Kameng",
        3: "Papum Pare",
        4: "Upper Subansiri",
        5: "West Siang",
        6: "East Siang",
        7: "Upper Siang",
        8: "Changlang",
        9: "Tirap",
        10: "Lower Subansiri",
        11: "Kurung Kumey",
        12: "Dibang Valley",
        13: "Lower Dibang Valley",
        14: "Lohit",
        15: "Anjaw",
    },
    12: {
        0: "Mon",
        1: "Mokokchung",
        2: "Zunheboto",
        3: "Wokha",
        4: "Dimapur",
        5: "Phek",
        6: "Tuensang",
        7: "Longleng",
        8: "Kiphire",
        9: "Kohima",
        10: "Peren",
    },
    13: {
        0: "Senapati",
        1: "Tamenglong",
        2: "Churachandpur",
        3: "Bishnupur / Imphal East",
        4: "Thoubal",
        5: "Imphal West",
        7: "Ukhrul",
        8: "Chandel",
    },
    14: {
        0: "Mamit",
        1: "Kolasib",
        2: "Aizwal",
        3: "Champhai",
        4: "Serchhip",
        5: "Lunglei",
        6: "Lawngtlai",
        7: "Saiha",
    },
    15: {
        0: "West Tripura",
        1: "South Tripura",
        2: "Dhalai",
        3: "North Tripura",
    },
    16: {
        0: "West Garo Hills",
        1: "East Garo Hills",
        2: "South Garo Hills",
        3: "West Khasi Hills",
        4: "Ribhoi",
        5: "East Khasi Hills",
        6: "Jaintia Hills",
    },
    17: {
        0: "Kokrajhar",
        1: "Dhubri",
        2: "Goalpara",
        3: "Barpeta",
        4: "Morigaon",
        5: "Nagaon",
        6: "Sonitpur",
        7: "Lakhimpur",
        8: "Dhemaji",
        9: "Tinsukia",
        10: "Dibrugarh",
        11: "Sivasagar",
        12: "Jorhat",
        13: "Golaghat",
        14: "Karbi Anglong",
        15: "Dima Hasao",
        16: "Cachar",
        17: "Karimganj",
        18: "Hailakandi",
        19: "Bongaigaon",
        20: "Chirang",
        21: "Kamrup",
        22: "Kamrup Metropolitan",
        23: "Nalbari",
        24: "Baksa",
        25: "Darrang",
        26: "Udalguri",
    },
    18: {
        0: "Darjiling",
        1: "Jalpaiguri",
        2: "Koch Bihar",
        3: "Uttar Dinajpur",
        4: "Dakshin Dinajpur",
        5: "Maldah",
        6: "Murshidabad",
        7: "Birbhum",
        8: "Purba Barddhaman",
        9: "Nadia",
        10: "North Twenty Four Parganas",
        11: "Hugli",
        12: "Bankura",
        13: "Puruliya",
        14: "Haora",
        15: "Kolkata",
        16: "South Twenty Four Parganas",
        17: "Paschim Medinipur",
        18: "Purba Medinipur",
        19: "Alipurduar",
        20: "Kalimpong",
        21: "Jhargram",
        22: "Paschim Barddhaman",
    },
    19: {
        0: "Garhwa",
        1: "Chatra",
        2: "Kodarma",
        3: "Giridih",
        4: "Deoghar",
        5: "Godda",
        6: "Sahibganj",
        7: "Pakur",
        8: "Dhanbad",
        9: "Bokaro",
        10: "Lohardaga",
        11: "Purbi Singhbhum",
        12: "Palamu",
        13: "Latehar",
        14: "Hazaribagh",
        15: "Ramgarh",
        16: "Dumka",
        17: "Jamtara",
        18: "Ranchi",
        19: "Khunti",
        20: "Gumla",
        21: "Simdega",
        22: "Pashchimi Singhbhum",
        23: "Saraikela-Kharsawan",
    },
    20: {
        0: "Bargarh",
        1: "Jharsuguda",
        2: "Sambalpur",
        3: "Debagarh",
        4: "Sundargarh",
        5: "Kendujhar",
        6: "Mayurbhanj",
        7: "Baleshwar",
        8: "Bhadrak",
        9: "Kendrapara",
        10: "Jagatsinghapur",
        11: "Cuttack",
        12: "Jajapur",
        13: "Dhenkanal",
        14: "Anugul",
        15: "Nayagarh",
        16: "Khordha",
        17: "Puri",
        18: "Ganjam",
        19: "Gajapati",
        20: "Kandhamal",
        21: "Baudh",
        22: "Subarnapur",
        23: "Balangir",
        24: "Nuapada",
        25: "Kalahandi",
        26: "Rayagada",
        27: "Nabarangapur",
        28: "Koraput",
        29: "Malkangiri",
    },
    21: {
        0: "Koriya",
        1: "Surguja",
        2: "Jashpur",
        3: "Raigarh",
        4: "Korba",
        5: "Janjgir-Champa",
        6: "Bilaspur",
        7: "Kabeerdham",
        8: "Rajnandgaon",
        9: "Durg",
        10: "Raipur",
        11: "Mahasamund",
        12: "Dhamtari",
        13: "Uttar Bastar Kanker",
        14: "Bastar",
        15: "Narayanpur",
        16: "Dakshin Bastar Dantewada",
        17: "Bijapur",
        18: "Balodabazar",
        19: "Gariyaband",
        20: "Kondagaon",
        21: "Sukama",
        22: "Bemetara",
        23: "Balod",
        24: "Mungeli",
        25: "Surajpur",
        26: "Balrampur",
    },
    22: {
        0: "Sheopur",
        1: "Morena",
        2: "Bhind",
        3: "Gwalior",
        4: "Datia",
        5: "Shivpuri",
        6: "Tikamgarh",
        7: "Chhatarpur",
        8: "Panna",
        9: "Sagar",
        10: "Damoh",
        11: "Satna",
        12: "Rewa",
        13: "Umaria",
        14: "Neemuch",
        15: "Mandsaur",
        16: "Ratlam",
        17: "Ujjain",
        18: "Shajapur",
        19: "Dewas",
        20: "Dhar",
        21: "Indore",
        22: "Khargone (West Nimar)",
        23: "Barwani",
        24: "Rajgarh",
        25: "Vidisha",
        26: "Bhopal",
        27: "Sehore",
        28: "Raisen",
        29: "Betul",
        30: "Harda",
        31: "Hoshangabad",
        32: "Katni",
        33: "Jabalpur",
        34: "Narsimhapur",
        35: "Dindori",
        36: "Mandla",
        37: "Chhindwara",
        38: "Seoni",
        39: "Balaghat",
        40: "Guna",
        41: "Ashoknagar",
        42: "Shahdol",
        43: "Anuppur",
        44: "Sidhi",
        45: "Singrauli",
        46: "Jhabua",
        47: "Alirajpur",
        48: "Khandwa (East Nimar)",
        49: "Burhanpur",
    },
    23: {
        0: "Kachchh",
        1: "Banas Kantha",
        2: "Patan",
        3: "Mahesana",
        4: "Sabar Kantha",
        5: "Gandhinagar",
        6: "Ahmadabad",
        7: "Surendranagar",
        8: "Rajkot",
        9: "Jamnagar",
        10: "Porbandar",
        11: "Junagadh",
        12: "Amreli",
        13: "Bhavnagar",
        15: "Anand",
        16: "Panch Mahals / Kheda",
        17: "Dohad",
        18: "Vadodara",
        19: "Narmada",
        20: "Bharuch",
        21: "The Dangs",
        22: "Navsari",
        23: "Valsad",
        24: "Surat",
        25: "Tapi",
        26: "Arvalli",
        27: "Botad",
        28: "Chhota Udepur",
        29: "Dev Bhumi-Dwarka",
        30: "Gir Somnath",
        31: "Mahisagar",
        32: "Morbi",
    },
    24: {
        0: "Diu",
        1: "Daman",
    },
    25: {
        0: "Dadra & Nagar Haveli",
    },
    26: {
        0: "Nandurbar",
        1: "Dhule",
        2: "Jalgaon",
        3: "Buldana",
        4: "Akola",
        5: "Washim",
        6: "Amravati",
        7: "Wardha",
        8: "Nagpur",
        9: "Bhandara",
        10: "Gondiya",
        11: "Gadchiroli",
        12: "Chandrapur",
        13: "Yavatmal",
        14: "Nanded",
        15: "Hingoli",
        16: "Parbhani",
        17: "Jalna",
        18: "Aurangabad",
        19: "Nashik",
        20: "Thane",
        21: "Mumbai Suburban",
        22: "Mumbai",
        23: "Raigarh",
        24: "Pune",
        25: "Ahmadnagar",
        26: "Bid",
        27: "Latur",
        28: "Osmanabad",
        29: "Solapur",
        30: "Satara",
        31: "Ratnagiri",
        32: "Sindhudurg",
        33: "Kolhapur",
        34: "Sangli",
    },
    27: {
        0: "Srikakulam",
        1: "Vizianagaram",
        2: "Visakhapatnam",
        3: "East Godavari",
        4: "West Godavari",
        5: "Krishna",
        6: "Guntur",
        7: "Prakasam",
        8: "Sri Potti Sriramulu Nellore",
        9: "Y.S.R. (Cuddapah)",
        10: "Kurnool",
        11: "Anantapur",
        12: "Chittoor",
    },
    28: {
        0: "Belgaum",
        1: "Bagalkot",
        2: "Bijapur",
        3: "Bidar",
        4: "Raichur",
        5: "Koppal",
        6: "Gadag",
        7: "Dharwad",
        8: "Uttara Kannada",
        9: "Haveri",
        10: "Bellary",
        11: "Chitradurga",
        12: "Davanagere",
        13: "Shimoga",
        14: "Udupi",
        15: "Chikmagalur",
        16: "Tumkur",
        17: "Bangalore",
        18: "Mandya",
        19: "Hassan",
        20: "Dakshina Kannada",
        21: "Kodagu",
        22: "Mysore",
        23: "Chamarajanagar",
        24: "Gulbarga",
        25: "Yadgir",
        26: "Kolar",
        27: "Chikkaballapura",
        28: "Bangalore Rural",
        29: "Ramanagara",
    },
    29: {
        0: "North Goa",
        1: "South Goa",
    },
    30: {
        0: "Lakshadweep",
    },
    31: {
        0: "Kasaragod",
        1: "Kannur",
        2: "Wayanad",
        3: "Kozhikode",
        4: "Malappuram",
        5: "Palakkad",
        6: "Thrissur",
        7: "Ernakulam",
        8: "Idukki",
        9: "Kottayam",
        10: "Alappuzha",
        11: "Pathanamthitta",
        12: "Kollam",
        13: "Thiruvananthapuram",
    },
    32: {
        0: "Thiruvallur",
        1: "Chennai",
        2: "Kancheepuram",
        3: "Vellore",
        4: "Tiruvannamalai",
        5: "Viluppuram",
        6: "Salem",
        7: "Namakkal",
        8: "Erode",
        9: "The Nilgiris",
        10: "Dindigul",
        11: "Karur",
        12: "Tiruchirappalli",
        13: "Perambalur",
        14: "Ariyalur",
        15: "Cuddalore",
        16: "Nagapattinam",
        17: "Thiruvarur",
        18: "Thanjavur",
        19: "Pudukkottai",
        20: "Sivaganga",
        21: "Madurai",
        22: "Theni",
        23: "Virudhunagar",
        24: "Ramanathapuram",
        25: "Thoothukkudi",
        26: "Tirunelveli",
        27: "Kanniyakumari",
        28: "Dharmapuri",
        29: "Krishnagiri",
        30: "Coimbatore",
        31: "Tiruppur",
    },
    33: {
        0: "Yanam",
        1: "Puducherry",
        2: "Mahe",
        3: "Karaikal",
    },
    34: {
        0: "Nicobars",
        1: "North & Middle Andaman",
        2: "South Andaman",
    },
    35: {
        0: "Adilabad",
        1: "Komaram Bheem",
        2: "Mancherial",
        3: "Nirmal",
        4: "Nizamabad",
        5: "Jagtial",
        6: "Peddapalli",
        7: "Jayashankar",
        8: "Bhadradri",
        9: "Mahabubabad",
        10: "Warangal Rural",
        11: "Warangal Urban",
        12: "Karimnagar",
        13: "Rajanna",
        14: "Kamareddy",
        15: "Sangareddy",
        16: "Medak",
        17: "Siddipet",
        18: "Jangaon",
        19: "Yadadri",
        20: "Medchal-Malkajgiri",
        21: "Hyderabad",
        22: "Rangareddy",
        23: "Vikarabad",
        24: "Mahbubnagar",
        25: "Jogulamba",
        26: "Wanaparthy",
        27: "Nagarkurnool",
        28: "Nalgonda",
        29: "Suryapet",
        30: "Khammam",
    },
}
TIME_SLOTS = [
    "04:00",
    "04:30",
    "05:00",
    "05:30",
    "06:00",
    "06:30",
    "07:00",
    "07:30",
    "08:00",
    "08:30",
    "09:00",
    "09:30",
    "10:00",
    "10:30",
    "11:00",
    "11:30",
    "12:00",
    "12:30",
    "13:00",
    "13:30",
    "14:00",
    "14:30",
    "15:00",
    "15:30",
    "16:00",
    "16:30",
    "17:00",
    "17:30",
    "18:00",
    "18:30",
    "19:00",
    "19:30",
    "20:00",
    "20:30",
    "21:00",
    "21:30",
    "22:00",
    "22:30",
    "23:00",
    "23:30",
    "00:00",
    "00:30",
    "01:00",
    "01:30",
    "02:00",
    "02:30",
    "03:00",
    "03:30",
]

# ---------------------------------------------------------------------------
# Model & data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_resource
def load_model_and_data(version_key: str, ckpt_path: str, data_path: str):
    """Load the Generator checkpoint and the real-data NPZ.

    Returns (Generator, config_dict, npz_data) or raises an error.
    """
    if not os.path.exists(ckpt_path):
        st.error(f"Checkpoint not found at `{ckpt_path}`")
        st.stop()
    if not os.path.exists(data_path):
        st.error(f"Dataset NPZ not found at `{data_path}`")
        st.stop()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]

    data = np.load(data_path)
    num_districts = int(data["num_districts"])
    num_states = int(data["num_states"])

    # Read cond_dim from the actual dataset (checkpoint config may
    # have a stale default if it wasn't updated before saving).
    actual_cond_dim = data["cond_vector"].shape[1]

    Generator = get_generator_class(version_key)
    G = Generator(
        noise_dim=cfg["noise_dim"],
        cond_dim=actual_cond_dim,
        num_districts=num_districts,
        num_states=num_states,
        district_embed_dim=cfg["district_embed_dim"],
        state_embed_dim=cfg["state_embed_dim"],
        base_channels=cfg["g_base_channels"],
    )
    g_state_key = "G_state_ema" if "G_state_ema" in ckpt else "G_state"
    G.load_state_dict(ckpt[g_state_key])
    G.eval()

    # Update config with accurate value for downstream use
    cfg["cond_dim"] = actual_cond_dim

    return G, cfg, data


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def plot_timeline_strip(decoded: np.ndarray, title: str = "Activity Timeline"):
    """Horizontal colour-coded strip — one colour per 30-min slot."""
    fig, ax = plt.subplots(figsize=(14, 1.6))
    for i, code in enumerate(decoded):
        colour = ACTIVITY_COLORS.get(int(code), "#cccccc")
        ax.barh(0, 1, left=i, height=0.8, color=colour, edgecolor="white", linewidth=0.3)

    ax.set_xlim(0, 48)
    ax.set_yticks([])
    tick_positions = list(range(0, 48, 4))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([TIME_SLOTS[i] for i in tick_positions], fontsize=8, rotation=45)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Build legend
    patches = [
        mpatches.Patch(color=ACTIVITY_COLORS[k], label=ACTIVITY_LABELS[k])
        for k in sorted(ACTIVITY_LABELS)
    ]
    ax.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.55),
        ncol=3,
        fontsize=7,
        frameon=False,
    )

    plt.tight_layout()
    return fig


def plot_step_diary(decoded: np.ndarray, title: str = "Step Plot"):
    """Step plot showing activity transitions across time-slots."""
    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(48)
    ax.step(x, decoded, where="post", color="teal", linewidth=2)

    # Shade background by activity
    for i in range(48):
        colour = ACTIVITY_COLORS.get(int(decoded[i]), "#f0f0f0")
        ax.axvspan(i, i + 1, color=colour, alpha=0.18)

    ax.set_ylim(0.5, 9.5)
    ax.set_yticks(range(1, 10))
    ax.set_yticklabels([ACTIVITY_LABELS[k] for k in range(1, 10)], fontsize=8)
    tick_positions = list(range(0, 48, 4))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([TIME_SLOTS[i] for i in tick_positions], rotation=45, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def time_breakdown_table(decoded: np.ndarray):
    """Return a list-of-dicts with minutes per activity."""
    rows = []
    for code in sorted(ACTIVITY_LABELS):
        count = int((decoded == code).sum())
        minutes = count * 30
        pct = count / 48 * 100
        rows.append(
            {
                "Activity": ACTIVITY_LABELS[code],
                "Slots (30 min)": count,
                "Minutes": minutes,
                "% of Day": f"{pct:.1f}%",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_generate(G, cfg, data):
    """Generate Diary page."""
    st.header("Generate Synthetic Daily Routine")

    # --- Sidebar controls ---
    st.sidebar.header("Demographics")

    age_sel = st.sidebar.selectbox("Age Group", AGE_LABELS, index=3)
    gender_sel = st.sidebar.selectbox("Gender", GENDER_LABELS, index=0)
    marital_sel = st.sidebar.selectbox("Marital Status", MARITAL_LABELS, index=0)
    edu_sel = st.sidebar.selectbox("Education Level", list(EDU_LABELS.values()), index=8)
    act_sel = st.sidebar.selectbox("Principal Activity", list(ACT_LABELS.values()), index=3)
    dow_sel = st.sidebar.selectbox("Day of Week", DOW_LABELS, index=0)
    sector_sel = st.sidebar.radio("Sector", SECTOR_LABELS, index=1)
    caregiving_sel = st.sidebar.checkbox("Caregiving Required", value=False)

    num_districts = int(data["num_districts"])
    num_states = int(data["num_states"])

    # --- State selection (named) ---
    state_options = {
        sid: STATE_NAMES.get(sid, f"State {sid}")
        for sid in range(num_states)
    }
    state_sel_idx = st.sidebar.selectbox(
        "State",
        range(num_states),
        index=min(1, num_states - 1),
        format_func=lambda sid: f"[{sid}] {state_options[sid]}",
    )
    state_id = state_sel_idx

    # --- District selection (filtered by selected state) ---
    state_district_map = DISTRICT_NAMES_BY_STATE.get(state_id, {})
    # Build list of district IDs that exist in data for this state
    all_district_ids = list(range(num_districts))
    if state_district_map:
        # Show only districts that belong to this state (from PDF mapping)
        available_district_ids = sorted(state_district_map.keys())
    else:
        # Fallback: show all district IDs if no mapping exists
        available_district_ids = all_district_ids

    district_options = {
        did: state_district_map.get(did, f"District {did}")
        for did in available_district_ids
    }
    district_sel_idx = st.sidebar.selectbox(
        "District",
        available_district_ids,
        index=0,
        format_func=lambda did: f"[{did}] {district_options[did]}",
    )
    district_id = district_sel_idx

    # --- Gumbel-Softmax Parameters (v3 Feature!) ---
    st.sidebar.header("Gumbel-Softmax Settings")
    gumbel_temp = st.sidebar.slider("Gumbel Temperature", 0.05, 2.0, 0.1, 0.05)
    gumbel_hard = st.sidebar.checkbox("Hard Discretization", value=True)

    num_samples = st.sidebar.number_input("Number of diaries", min_value=1, max_value=100, value=10)

    generate = st.sidebar.button("Generate Diary", type="primary", use_container_width=True)

    # --- Info blurb ---
    st.info(
        "**How it works:** A demographic conditioning vector is combined with your custom "
        "District & State parameters. The Generator outputs categorical activity logits, which are discretised "
        "using **Gumbel-Softmax** (controlled by the Temperature & Hard Discretization settings on the left) "
        "into a crisp 24-hour routine."
    )

    if not generate:
        return

    # --- Build conditioning vector ---
    all_cond = data["cond_vector"]  # (N, cond_dim)
    n_real = all_cond.shape[0]

    for sample_idx in range(num_samples):
        # Pick a random real sample as template
        rand_idx = np.random.randint(0, n_real)
        cond_vec = all_cond[rand_idx].copy()  # (cond_dim,)

        # Inference
        with torch.no_grad():
            z = torch.randn(1, cfg["noise_dim"])
            cv = torch.from_numpy(cond_vec).float().unsqueeze(0)
            di = torch.tensor([district_id]).long()
            si = torch.tensor([state_id]).long()

            # Pass temp and hard options dynamically to v3 Generator
            fake = G(z, cv, di, si, temp=gumbel_temp, hard=gumbel_hard)  # (1, 9, 48, 1)
            fake_np = fake.squeeze(-1).squeeze(0).numpy()  # (9, 48)

        # Decode: argmax across 9 channels → activity code 1-9
        decoded = np.argmax(fake_np, axis=0) + 1  # (48,)

        # --- Display ---
        if num_samples > 1:
            st.subheader(f"Sample {sample_idx + 1}")

        # 1) Colour-coded timeline strip
        title_strip = f"Timeline — {gender_sel}, {age_sel}"
        fig_strip = plot_timeline_strip(decoded, title=title_strip)
        st.pyplot(fig_strip)
        plt.close(fig_strip)

        # 2) Step plot
        fig_step = plot_step_diary(decoded, title="Activity Step Plot")
        st.pyplot(fig_step)
        plt.close(fig_step)

        # 3) Time breakdown table
        st.subheader("⏱️ Time Breakdown")
        breakdown = time_breakdown_table(decoded)
        st.table(breakdown)

        st.divider()


def page_evaluation(eval_dir: str, version_label: str):
    """Show pre-computed evaluation images."""
    st.header(f"Model Evaluation Statistics ({version_label})")
    st.write(f"Comparison between Real ITUS 2019 data and {version_label} Synthetic data.")

    if not os.path.isdir(eval_dir):
        st.warning(
            f"Evaluation results directory `{eval_dir}` not found. "
            "Run evaluation scripts first to generate plots."
        )
        return

    # Auto-discover all PNG images in the evaluation directory
    images = sorted(glob.glob(os.path.join(eval_dir, "*.png")))
    if not images:
        st.warning(f"No PNG images found in `{eval_dir}`.")
        return

    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        nice_name = name.replace("_", " ").title()
        st.subheader(nice_name)
        st.image(img_path, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="TUS-GAN Dashboard",
        page_icon="📈",
        layout="wide",
    )
    st.title("TUS-GAN — Synthetic Time-Use Diary Generator")

    st.sidebar.title("Configuration")
    version = st.sidebar.selectbox(
        "Model Version", ["TUS-GAN v3 (Baseline)", "TUS-GAN v4", "TUS-GAN v5"], index=2
    )
    page = st.sidebar.selectbox("Navigation", ["Generate Diary", "Evaluation Results"])

    if "v3" in version:
        version_key = "v3"
        ckpt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "v3", "checkpoints", "final.pt"
        )
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "v3", "tusgan_encode.npz"
        )
        eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v3", "evaluation_results")
        version_label = "TUS-GAN v3"
    elif "v4" in version:
        version_key = "v4"
        ckpt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "v4", "checkpoints", "final.pt"
        )
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "v4", "tusgan_encode.npz"
        )
        eval_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "v4", "evaluation_results"
        )
        version_label = "TUS-GAN v4"
    else:
        version_key = "v5"
        ckpt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "v5", "checkpoints", "final.pt"
        )
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "v5", "tusgan_encode.npz"
        )
        eval_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "v5", "evaluation_v5_100k"
        )
        version_label = "TUS-GAN v5"

    G, cfg, data = load_model_and_data(version_key, ckpt_path, data_path)

    if page == "Generate Diary":
        page_generate(G, cfg, data)
    else:
        page_evaluation(eval_dir, version_label)


if __name__ == "__main__":
    main()
