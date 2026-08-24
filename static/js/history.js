const byId = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "").replace(
  /[&<>"']/g,
  (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[character],
);
const number = (value, digits = 3) => (
  Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "-"
);
const dateTime = (value) => (
  value ? new Date(value).toLocaleString() : "-"
);

let activeEventId = "";
let activeEvent = null;
let editingPiece = null;
const initialEventId = location.pathname.startsWith("/history/")
  ? decodeURIComponent(location.pathname.split("/").pop() || "")
  : "";

function measurement(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";

  let total = Math.round(Math.abs(numeric) * 16);
  const sign = numeric < 0 ? "-" : "";
  const feet = Math.floor(total / 192);
  total -= feet * 192;
  const inches = Math.floor(total / 16);
  let numerator = total % 16;
  let denominator = 16;
  while (numerator && numerator % 2 === 0) {
    numerator /= 2;
    denominator /= 2;
  }
  const inchText = numerator
    ? `${inches} ${numerator}/${denominator}`
    : `${inches}`;
  return `${sign}${feet}' ${inchText}"`;
}

function metric(label, value) {
  return `
    <div class="metric">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value ?? "-")}</strong>
    </div>`;
}

function renderPieces(pieces, databaseMode) {
  if (!pieces?.length) {
    return '<div class="empty">No pieces were stored for the canonical frame.</div>';
  }

  const table = `
    <table class="pieces">
      <thead>
        <tr>
          <th>Piece</th>
          <th>Automatic</th>
          <th>Operator</th>
          <th>Effective</th>
          <th>Difference</th>
          <th>Confidence</th>
          <th>Status</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        ${pieces.map((piece) => {
    const automatic = Number(piece.automatic_measurement_in);
    const operator = Number(piece.operator_measurement_in);
    const hasOperator = Number.isFinite(operator);
    const difference = hasOperator && Number.isFinite(automatic)
      ? measurement(operator - automatic)
      : "-";
    const state = piece.review_required || !piece.is_valid ? "review" : "ok";
    const editActions = databaseMode !== "simulation"
      ? `
        <button class="btn edit-piece" data-piece-id="${escapeHtml(piece.id)}">Edit</button>
        ${hasOperator ? `<button class="btn danger clear-piece" data-piece-id="${escapeHtml(piece.id)}">Clear</button>` : ""}`
      : "";

    return `
          <tr>
            <td><strong>${escapeHtml(piece.piece_number)}</strong></td>
            <td>${escapeHtml(measurement(piece.automatic_measurement_in))}</td>
            <td>${escapeHtml(hasOperator ? measurement(piece.operator_measurement_in) : "-")}</td>
            <td><strong>${escapeHtml(measurement(piece.effective_measurement_in))}</strong></td>
            <td>${escapeHtml(difference)}</td>
            <td>YOLO ${number(piece.yolo_confidence, 2)} / Edge ${number(piece.sobel_confidence, 2)}</td>
            <td>
              <span class="state ${state}">${state === "ok" ? "Valid" : "Review"}</span>
              <br>
              <span class="muted">rev ${escapeHtml(piece.operator_revision ?? 0)}</span>
            </td>
            <td>${editActions}</td>
          </tr>`;
  }).join("")}
      </tbody>
    </table>`;

  const revisions = pieces.flatMap((piece) => (
    (piece.revisions || []).map((revision) => ({ piece, revision }))
  ));
  if (!revisions.length) return table;

  return `${table}
    <div class="revision-log">
      ${revisions.map(({ piece, revision }) => `
        <div class="revision">
          <strong>Piece ${escapeHtml(piece.piece_number)} / rev ${escapeHtml(revision.revision)}</strong>
          <span>${escapeHtml(revision.action)} | ${escapeHtml(dateTime(revision.changed_at))}</span>
          <span>${escapeHtml(revision.operator_display_name || revision.operator_id)}:
            ${escapeHtml(revision.reason)}
            (${escapeHtml(measurement(revision.previous_operator_measurement_in))}
            to ${escapeHtml(measurement(revision.new_operator_measurement_in))})
          </span>
        </div>`).join("")}
    </div>`;
}

function renderSnapshots(snapshots) {
  if (!snapshots?.length) {
    return '<div class="empty">No processing evidence was stored.</div>';
  }

  return `
    <div class="snapshots">
      ${snapshots.map((snapshot) => {
    const summary = snapshot.measurement_summary || {};
    const image = snapshot.original_overlay_url
      ? `<img class="measurement-evidence" src="${escapeHtml(snapshot.original_overlay_url)}" alt="Measurement evidence">`
      : '<div class="empty">No camera evidence</div>';

    return `
        <article class="snapshot">
          <div class="snapshot-head">
            <span>${escapeHtml(dateTime(snapshot.processed_utc || snapshot.frame_utc))}</span>
            <span>${escapeHtml(summary.valid_count ?? snapshot.valid_piece_count ?? 0)}
              valid of ${escapeHtml(summary.detected_count ?? snapshot.detected_piece_count ?? 0)}
            </span>
          </div>
          <div class="snapshot-grid">${image}</div>
        </article>`;
  }).join("")}
    </div>`;
}

function bindPieceActions() {
  document.querySelectorAll(".edit-piece").forEach((button) => {
    button.addEventListener(
      "click",
      () => openCorrection(button.dataset.pieceId),
    );
  });
  document.querySelectorAll(".clear-piece").forEach((button) => {
    button.addEventListener(
      "click",
      () => clearCorrection(button.dataset.pieceId),
    );
  });
}

async function loadEvent(eventId) {
  activeEventId = eventId;
  document.querySelectorAll(".event").forEach((element) => {
    element.classList.toggle(
      "active",
      element.dataset.eventId === eventId,
    );
  });

  const response = await fetch(
    `/api/history/events/${encodeURIComponent(eventId)}`,
  );
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Could not load measurement event");
  }

  activeEvent = data;
  byId("detail-title").textContent = dateTime(
    data.plc_source_timestamp || data.app_received_at || data.created_at,
  );
  byId("detail-state").textContent = data.status || "";
  byId("viewer").innerHTML = `
    ${data.video_url
    ? `<video controls preload="metadata" src="${escapeHtml(data.video_url)}"></video>`
    : '<div class="empty">No video asset is available.</div>'}
    ${data.raw_video_url
    ? `<div class="asset-actions"><a class="btn" href="${escapeHtml(data.raw_video_url)}" target="_blank" rel="noopener">Raw clip</a></div>`
    : ""}
    <div class="meta">
      ${metric("PLC signal", dateTime(data.plc_source_timestamp || data.app_received_at))}
      ${metric("Recording", `${dateTime(data.recording_started_at)} - ${dateTime(data.recording_ended_at)}`)}
      ${metric("Pieces", `${data.valid_piece_count ?? 0} valid / ${data.detected_piece_count ?? 0} detected`)}
      ${metric("Status", data.status || "-")}
    </div>
    <div class="section-head">
      <h3>Piece measurements</h3>
      <span class="muted">Canonical processed frame</span>
    </div>
    ${renderPieces(data.pieces, data.database_mode)}
    <div class="section-head">
      <h3>Processing evidence</h3>
      <span class="muted">PLC signal measurement frame</span>
    </div>
    ${renderSnapshots(data.snapshots)}
  `;
  bindPieceActions();
}

async function loadHistory() {
  const response = await fetch("/api/history/events?limit=100");
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Could not load history");
  }

  byId("event-count").textContent = `${data.count} saved events`;
  if (!data.events.length) {
    byId("event-list").innerHTML = (
      '<div class="empty">No PLC events have been saved.</div>'
    );
    return;
  }

  byId("event-list").innerHTML = data.events.map((event) => `
    <button class="event" data-event-id="${escapeHtml(event.id)}">
      <strong>${escapeHtml(dateTime(event.plc_source_timestamp || event.app_received_at || event.created_at))}</strong>
      <span>${escapeHtml(event.valid_piece_count ?? 0)} valid /
        ${escapeHtml(event.detected_piece_count ?? 0)} detected |
        ${escapeHtml(event.status || "-")}
      </span>
    </button>
  `).join("");
  document.querySelectorAll(".event").forEach((button) => {
    button.addEventListener(
      "click",
      () => loadEvent(button.dataset.eventId),
    );
  });

  const initial = (
    data.events.find((event) => String(event.id) === initialEventId)
    || data.events[0]
  );
  await loadEvent(String(initial.id));
}

function openCorrection(pieceId) {
  editingPiece = activeEvent?.pieces?.find(
    (piece) => String(piece.id) === String(pieceId),
  );
  if (!editingPiece) return;

  const total = Number(
    editingPiece.operator_measurement_in
    ?? editingPiece.automatic_measurement_in
    ?? 0,
  );
  let units = Math.max(0, Math.round(total * 16));
  byId("feet").value = Math.floor(units / 192);
  units %= 192;
  byId("inches").value = Math.floor(units / 16);
  byId("sixteenths").value = units % 16;
  byId("operator-id").value = localStorage.getItem("tx2OperatorId") || "";
  byId("operator-name").value = localStorage.getItem("tx2OperatorName") || "";
  byId("reason").value = "";
  byId("form-error").textContent = "";
  byId("correction-title").textContent = (
    `Piece ${editingPiece.piece_number} operator measurement`
  );
  byId("correction-dialog").showModal();
}

async function clearCorrection(pieceId) {
  const piece = activeEvent?.pieces?.find(
    (item) => String(item.id) === String(pieceId),
  );
  if (
    !piece
    || !confirm(`Clear the operator correction for piece ${piece.piece_number}?`)
  ) {
    return;
  }

  const operatorId = localStorage.getItem("tx2OperatorId") || "";
  const reason = prompt("Reason for clearing this correction:") || "";
  if (!operatorId || !reason) {
    alert(
      "Operator ID and reason are required. Open Edit once to store the operator ID.",
    );
    return;
  }
  await patchCorrection(
    piece,
    {
      clear: true,
      reason,
      operator_id: operatorId,
      operator_display_name: (
        localStorage.getItem("tx2OperatorName") || null
      ),
    },
  );
}

async function patchCorrection(piece, payload) {
  const response = await fetch(
    `/api/history/events/${encodeURIComponent(activeEventId)}/pieces/${encodeURIComponent(piece.id)}/operator-measurement`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...payload,
        expected_revision: piece.operator_revision ?? 0,
      }),
    },
  );
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Could not save correction");
  }
  await loadEvent(activeEventId);
}

byId("correction-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const operatorId = byId("operator-id").value.trim();
  const operatorName = byId("operator-name").value.trim();
  localStorage.setItem("tx2OperatorId", operatorId);
  localStorage.setItem("tx2OperatorName", operatorName);

  try {
    await patchCorrection(
      editingPiece,
      {
        feet: Number(byId("feet").value),
        inches: Number(byId("inches").value),
        sixteenths: Number(byId("sixteenths").value),
        operator_id: operatorId,
        operator_display_name: operatorName || null,
        reason: byId("reason").value.trim(),
      },
    );
    byId("correction-dialog").close();
  } catch (error) {
    byId("form-error").textContent = error.message || error;
  }
});

byId("cancel-correction").addEventListener(
  "click",
  () => byId("correction-dialog").close(),
);

loadHistory().catch((error) => {
  byId("event-list").innerHTML = (
    `<div class="empty">${escapeHtml(error.message || error)}</div>`
  );
});
