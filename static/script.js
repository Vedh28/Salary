
document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("salaryForm");
    const button = document.getElementById("predictBtn");
    const errorBox = document.getElementById("predictionError");
    const resultBox = document.getElementById("resultBox");
    const salaryResult = document.getElementById("salaryResult");
    const resultMessage = document.getElementById("resultMessage");

    // Salary prediction form
    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        errorBox.textContent = "";
        resultBox.hidden = true;
        button.disabled = true;
        button.textContent = "Calculating...";

        try {
            const payload = {
                age: document.getElementById("age").value,
                experience: document.getElementById("experience").value,
                education: document.getElementById("education").value,
                job: document.getElementById("job").value
            };

            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || "Prediction failed.");
            }

            const salary = Number(result.predicted_salary);

            if (!Number.isFinite(salary)) {
                throw new Error("Invalid salary received from server.");
            }

            salaryResult.textContent =
                "₹" + salary.toLocaleString("en-IN", {
                    maximumFractionDigits: 0
                });

            resultMessage.textContent =
                "Estimated by " + result.model;

            resultBox.hidden = false;
            resultBox.scrollIntoView({
                behavior: "smooth",
                block: "nearest"
            });

        } catch (error) {
            errorBox.textContent =
                error.message || "Something went wrong.";
        } finally {
            button.disabled = false;
            button.innerHTML = 'Predict my salary <span>→</span>';
        }
    });

    // Dashboard chart data comes from Flask
    const data = window.salaryDashboardData || {};
    const chartError = document.getElementById("chartError");

    if (typeof Chart === "undefined") {
        chartError.textContent =
            "Charts could not load. Check your internet connection.";
        return;
    }

    try {
        const axisStyle = {
            ticks: { color: "#9caac5" },
            grid: { color: "#27334d" }
        };

        const commonOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: "#d7e0f4" }
                }
            },
            scales: {
                x: { ...axisStyle },
                y: { ...axisStyle }
            }
        };

        // Chart 1: Average salary by job
        new Chart(document.getElementById("salaryChart"), {
            type: "bar",
            data: {
                labels: data.jobLabels,
                datasets: [{
                    label: "Average annual salary (₹)",
                    data: data.jobAverages,
                    backgroundColor: [
                        "#9c83ff",
                        "#55e5d0",
                        "#7c9dff",
                        "#f2b880",
                        "#db8bff",
                        "#6ed2ff"
                    ],
                    borderRadius: 7
                }]
            },
            options: commonOptions
        });

        // Chart 2: Salary distribution
        new Chart(document.getElementById("distributionChart"), {
            type: "line",
            data: {
                labels: data.salaries.map((_, index) => index + 1),
                datasets: [{
                    label: "Salary (₹)",
                    data: data.salaries,
                    borderColor: "#55e5d0",
                    backgroundColor: "#55e5d022",
                    fill: true,
                    pointRadius: 0,
                    borderWidth: 2
                }]
            },
            options: commonOptions
        });

        // Chart 3: Actual vs predicted salary
        new Chart(document.getElementById("actualChart"), {
            type: "scatter",
            data: {
                datasets: [{
                    label: "Actual vs predicted",
                    data: data.actualPredicted,
                    backgroundColor: "#9c83ff",
                    pointRadius: 4
                }]
            },
            options: {
                ...commonOptions,
                scales: {
                    x: {
                        ...axisStyle,
                        type: "linear",
                        title: {
                            display: true,
                            text: "Actual salary (₹)",
                            color: "#d7e0f4"
                        }
                    },
                    y: {
                        ...axisStyle,
                        title: {
                            display: true,
                            text: "Predicted salary (₹)",
                            color: "#d7e0f4"
                        }
                    }
                }
            }
        });

        // Chart 4: Prediction residuals
        new Chart(document.getElementById("residualChart"), {
            type: "bar",
            data: {
                labels: data.residuals.map((_, index) => index + 1),
                datasets: [{
                    label: "Prediction error (₹)",
                    data: data.residuals,
                    backgroundColor: "#f2b880",
                    borderRadius: 3
                }]
            },
            options: commonOptions
        });

    } catch (error) {
        console.error("Dashboard chart error:", error);
        chartError.textContent =
            "Charts could not be displayed. Refresh the page.";
    }
});
