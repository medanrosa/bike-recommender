document.getElementById("criteria-form").addEventListener("submit", async e => {
  e.preventDefault();
  // gather inputs…
  const payload = { experience_level: 0, preferred_style: 2, geography: 4, budget: 7000 };
  const res = await fetch("/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  document.getElementById("results").innerText =
    data.error ? data.error : JSON.stringify(data, null, 2);
});