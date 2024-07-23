# Install and load the plotly package
# install.packages("plotly")
library(plotly)

# Create the data frame
data <- data.frame(
  Location = c("A", "B", "C", "D", "E"),
  Temperature = c(15, 20, 18, 12, 17),
  Humidity = c(65, 70, 68, 60, 72),
  CO2 = c(400, 450, 420, 380, 430)
)

# Create a grid of temperature values
temp_seq <- seq(min(data$Temperature), max(data$Temperature), length.out = 50)
humidity_val <- mean(data$Humidity)  # Use the mean humidity for this plot
grid_temp <- expand.grid(Temperature = temp_seq, Humidity = humidity_val)

# Fit a linear model to predict CO2 based on temperature and humidity
lm_temp <- lm(CO2 ~ Temperature, data = data)

# Predict CO2 levels for the grid of temperature values
grid_temp$CO2 <- predict(lm_temp, newdata = grid_temp)

# Create the 3D surface plot for CO2 vs. Temperature
plot_ly(grid_temp, x = ~Temperature, y = ~Humidity, z = ~CO2, type = "surface") %>%
  layout(title = "3D Surface Plot: CO2 Levels vs. Temperature (Mean Humidity)",
         scene = list(
           xaxis = list(title = 'Temperature (°C)'),
           yaxis = list(title = 'Humidity (Mean)'),
           zaxis = list(title = 'CO2 Levels (ppm)')
         ))
