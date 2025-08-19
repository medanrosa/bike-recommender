document.getElementById("criteria-form")?.addEventListener("submit", async (e) => {
  e.preventDefault();

  // Example payload WITHOUT preferred_style (geography decides style)
  const payload = {
    experience_level: 0,
    geography: 4,
    budget: 7000
  };

  const res = await fetch("/recommend", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload)
  });

  const data = await res.json();
  document.getElementById("results").innerText =
    data.error ? data.error : JSON.stringify(data, null, 2);
});