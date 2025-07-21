{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww26040\viewh14940\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 from nemo.collections.llm.api import LLM\
\
# Assuming you have already initialized your LLM\
# Replace with your actual model initialization\
llm = LLM(...)  # Initialize your LLM instance\
\
sampling_params = \{\
    "max_length": 100  # Set the maximum length to 100 tokens\
\}\
\
# Generate text with the specified sampling parameters\
response = llm.generate(prompt="Your prompt here", sampling_params=sampling_params)\
\
print(response)}