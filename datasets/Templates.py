templates = [
    # --- Temperature Maximum - Annual ---
    {
        "user": "What is the projected annual max temperature at {grid} by the end of the century?",
        "assistant": "At grid {grid}, the annual maximum temperature is projected to reach {temp}°F under RCP 8.5 by the end of the century.",
    },
    {
        "user": "How does the max annual temperature increase at {grid} compare to nearby grid cells?",
        "assistant": "Grid {grid} is projected to warm by {change}°F, placing it in the {state_percentile} percentile within the state for temperature increases.",
    },
    # --- Temperature Minimum - Seasonal ---
    {
        "user": "What is the projected change in winter minimum temperature at {grid}?",
        "assistant": "The winter minimum temperature at {grid} is expected to increase by {change}°F, which is considered {relative_value} relative to the rest of the state.",
    },
    {
        "user": "How does the projected seasonal min temperature change at {grid} compare nationally?",
        "assistant": "Grid {grid} ranks in the {national_percentile} percentile for minimum seasonal temperature rise under RCP 8.5, suggesting above-average warming.",
    },
    # --- Precipitation – Annual Total ---
    {
        "user": "What is the projected change in annual precipitation at {grid}?",
        "assistant": "Annual precipitation at {grid} is projected to increase by {change} inches, with variability depending on the scenario.",
    },
    {
        "user": "Compare total annual rainfall at {grid} to the rest of the region.",
        "assistant": "With {value} inches of projected rainfall, grid {grid} falls in the {state_percentile} percentile compared to other areas in the state.",
    },
    # --- Precipitation None – Annual Average ---
    {
        "user": "How many dry days per year are projected for {grid}?",
        "assistant": "Grid {grid} is projected to have an average of {value} dry days annually, which is {trend_description} compared to current climate patterns.",
    },
    # --- Wind Speed – Annual Average ---
    {
        "user": "What is the expected average wind speed at {grid}?",
        "assistant": "The projected average wind speed at grid {grid} is {value} mph, consistent with the regional trend.",
    },
    {
        "user": "How does wind speed at {grid} compare to nearby areas?",
        "assistant": "Grid {grid} falls in the {state_percentile} percentile for wind speed, indicating it is {wind_trend} windier than most grid cells in the state.",
    },
    # --- Cooling Degree Days – Annual Total ---
    {
        "user": "How are cooling needs changing at {grid}?",
        "assistant": "Cooling degree days at grid {grid} are expected to increase by {change}, reflecting a greater need for air conditioning under future climate scenarios.",
    },
    # --- Heating Degree Days – Annual Total ---
    {
        "user": "How is the need for winter heating projected to change at {grid}?",
        "assistant": "Heating degree days are projected to decrease at {grid}, with a reduction of {change} days annually, implying milder winters.",
    },
    {
        "user": "What are the implications of reduced heating days at {grid}?",
        "assistant": "With {change} fewer heating degree days, grid {grid} may see lower winter energy demands relative to historical norms.",
    },
    # --- Fire Weather Index – Averages ---
    {
        "user": "What is the projected fire weather index at {grid}?",
        "assistant": "The average fire weather index (FWI) at grid {grid} is projected to be {value}, placing it in the {state_percentile} percentile statewide.",
    },
    {
        "user": "How does fire risk at {grid} compare to national trends?",
        "assistant": "Grid {grid} has an FWI percentile of {national_percentile} under RCP 8.5, indicating {risk_description} relative fire weather risk nationally.",
    },
    # --- Fire Weather Index – FWI Classes ---
    {
        "user": "How frequently does {grid} experience extreme fire conditions?",
        "assistant": "Grid {grid} is projected to experience {value} days per year in the 'Extreme' FWI class by the end of the century.",
    },
    # --- Heat Index – Summer Seasonal ---
    {
        "user": "What will summer feel like at {grid} in the future?",
        "assistant": "The summer heat index at {grid} is projected to rise to {value}°F, with more frequent days exceeding heat advisory thresholds.",
    },
    {
        "user": "How is summer heat stress expected to change at {grid}?",
        "assistant": "Summer heat stress is expected to worsen at grid {grid}, with the seasonal heat index increasing by {change}°F under RCP 8.5.",
    },
    # --- General Comparative ---
    {
        "user": "Which variables show the biggest projected changes at {grid}?",
        "assistant": "Grid {grid} is expected to see major changes in temperature max (+{temp_change}°F), fire weather index (+{fwi_change}), and cooling degree days (+{cdd_change}).",
    },
    {
        "user": "Summarize the future climate trends for {grid}.",
        "assistant": "At grid {grid}, projections show higher summer heat index ({heat_index}), more dry days ({dry_days}), and a rise in annual max temperature by {temp_change}°F.",
    },
    # --- Relative Change Interpretation (with risk framing) ---
    {
        "user": "How much has the number of days above 95°F changed at {grid} from historical to mid-century?",
        "assistant": "Grid {grid} is projected to experience an increase from {hist_days} to {mid_days} days above 95°F by mid-century, a {percent_change}% increase, tripling the number of dangerous heat days.",
    },
    {
        "user": "What does the change in heat index days at {grid} suggest about future conditions?",
        "assistant": "Although the increase at grid {grid} is only from {hist_days} to {mid_days} days, that's a {percent_change}% rise, indicating a substantial change in heat stress conditions.",
    },
    # --- Climate Risk Framing ---
    {
        "user": "What are the risks posed by projected climate changes at {grid}?",
        "assistant": "Grid {grid} shows increased risk of heat-related stress with a rise in days above 95°F and reduced precipitation. These changes can intensify drought and cooling energy demands.",
    },
    {
        "user": "How might changes at {grid} affect farming or housing decisions?",
        "assistant": "Rising maximum temperatures and fewer rainy days at grid {grid} could impact crop resilience and water supply. Higher heat index values may also influence building codes and energy needs.",
    },
    # --- Variable Interaction Insight ---
    {
        "user": "What other variables should I look at along with heat index at {grid}?",
        "assistant": "To understand the full impact, consider the increase in cooling degree days and dry days alongside the heat index at grid {grid}. These variables together suggest greater cooling demand and heat stress risk.",
    },
    {
        "user": "What factors contribute to concern about rising heat index at {grid}?",
        "assistant": "At grid {grid}, the increase in heat index is compounded by rising temperatures and lower wind speeds, reducing the body’s ability to cool and raising the chance of heat-related illnesses.",
    },
    # --- Cross-Region Comparison ---
    {
        "user": "How does the heat risk at {grid} compare to other areas in the state?",
        "assistant": "Grid {grid} ranks in the top {state_percentile}% for days above 95°F, indicating one of the highest heat exposure risks in the state.",
    },
    {
        "user": "How does wind speed at {grid} compare to other regions?",
        "assistant": "Grid {grid} has lower wind speeds than {comparison_region}, which may reduce natural ventilation and increase the severity of heat stress events.",
    },
    # --- Neighborhood/Adjacency Context ---
    {
        "user": "How does projected warming at {grid} compare to surrounding grid cells?",
        "assistant": "Grid {grid} has a projected maximum temperature increase of {change}°F, higher than its neighboring cells, making it a local hotspot under RCP 8.5.",
    },
    {
        "user": "How does fire risk at {grid} compare to nearby areas?",
        "assistant": "Grid {grid} lies in the {state_percentile} percentile for fire weather index, indicating elevated risk compared to adjacent areas in the same region.",
    },
    # --- Usability-Oriented Templates ---
    {
        "user": "I’m evaluating properties for purchase—what’s the future climate outlook at {grid}?",
        "assistant": "At grid {grid}, future projections show increased heat index days, more dry spells, and higher cooling energy needs. These factors could affect livability and long-term planning.",
    },
    {
        "user": "How could climate projections at {grid} affect grant proposals for urban infrastructure?",
        "assistant": "Grid {grid} shows substantial changes in temperature, precipitation, and heat stress indicators. These data support proposals focused on resilience, cooling access, and drought mitigation.",
    },
    {
        "user": "What extreme climate values are projected at {grid}?",
        "assistant": "Grid {grid} is in the top {national_percentile}% nationally for summer heat index, suggesting unusually intense seasonal heat.",
    },
    # --- State/Regional Summary & Extremes ---
    {
        "user": "Which state shows the largest projected increase in fire weather index?",
        "assistant": "Among all states, {state} shows the highest average projected increase in fire weather index under RCP 8.5, with values rising by {value}, reflecting significantly elevated wildfire risk.",
    },
    {
        "user": "What region in {state} is most affected by rising summer heat index?",
        "assistant": "Within {state}, the southern region has the highest projected summer heat index values, averaging {value}°F and placing many grid cells in the top {percentile} percentile nationally.",
    },
    {
        "user": "Can you summarize future climate shifts for {state}?",
        "assistant": "Projections for {state} indicate significant changes including a {temp_change}°F rise in annual max temperature, a {precip_change}% shift in annual rainfall, and an increase of {cdd_change} cooling degree days.",
    },
    {
        "user": "Which parts of {state} face the most fire risk in the future?",
        "assistant": "The western and central regions of {state} have grid cells in the top {state_percentile}% of fire weather index, signaling the highest concentration of projected fire risk.",
    },
    {
        "user": "What is the range of projected heat index values across {state} by 2100?",
        "assistant": "By the end of the century, summer heat index values in {state} are projected to range from {min_value}°F in the north to {max_value}°F in the southern regions, highlighting substantial intra-state variability.",
    },
    {
        "user": "Where in the U.S. are dry days projected to increase the most?",
        "assistant": "Regions in the Southwest, especially areas within {state}, are projected to experience the largest increase in dry days, with some grid cells gaining over {value} additional dry days annually.",
    },
    # --- Inter-State or Regional Comparison ---
    {
        "user": "How does {state1} compare to {state2} in terms of annual maximum temperature rise?",
        "assistant": "While both states show warming trends, {state1} has a higher projected increase in annual maximum temperature ({temp1}°F) compared to {state2} ({temp2}°F), suggesting more intense summer conditions.",
    },
    {
        "user": "Which state shows the biggest decline in precipitation by 2100?",
        "assistant": "{state} shows the largest projected decrease in annual precipitation, with totals declining by {value} inches, driven by fewer rainy days and increased seasonal dryness.",
    },
    {
        "user": "Is the projected change in fire risk more severe in {region1} or {region2}?",
        "assistant": "Fire weather index projections suggest {region1} faces higher average increases ({fwi1}) than {region2} ({fwi2}), placing it in a more vulnerable position by mid-century.",
    },
]
