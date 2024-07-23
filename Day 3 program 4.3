# Load necessary libraries
library(plotly)
library(cluster)

# Create the data frame
data <- data.frame(
  Location = c("A", "B", "C", "D", "E"),
  Temperature = c(15, 20, 18, 12, 17),
  Humidity = c(65, 70, 68, 60, 72),
  CO2 = c(400, 450, 420, 380, 430)
)

# Perform K-means clustering (choose an appropriate number of clusters)
set.seed(123)  # For reproducibility
kmeans_result <- kmeans(data[, c("Temperature", "Humidity", "CO2")], centers = 3)

# Add cluster results to the data frame
data$Cluster <- as.factor(kmeans_result$cluster)

# Create 3D scatter plot with clusters
plot_ly(data, x = ~Temperature, y = ~Humidity, z = ~CO2, color = ~Cluster, colors = c('red', 'green', 'blue'), type = "scatter3d", mode = "markers") %>%
  layout(title = "3D Scatter Plot with K-means Clustering",
         scene = list(
           xaxis = list(title = 'Temperature (°C)'),
           yaxis = list(title = 'Humidity (%)'),
           zaxis = list(title = 'CO2 Levels (ppm)')
         ))
