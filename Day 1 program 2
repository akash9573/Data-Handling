# Create a data frame
year_population_data <- data.frame(
  Year = 2010:2014,
  Population = c(1000, 1020, 1040, 1060, 1080)
)
# Plot line plot
ggplot(year_population_data, aes(x = Year, y = Population)) +
  geom_line(color = "green", size = 1) +
  geom_point(color = "red", size = 3) +
  labs(title = "Year vs. Population",
       x = "Year",
       y = "Population (millions)") +
  theme_minimal()
