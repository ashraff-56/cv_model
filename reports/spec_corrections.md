# Spec corrections and Phase 2 design notes (food recognition module)

These notes are ready to paste into the report. The spec text was not available, so each fix is keyed to the
section or diagram it changes rather than written as an exact find-and-replace.

## 1. One recognition component, named the same everywhere

**Problem.** The sequence diagram (5.4.3) names the model "YOLOv4 / MobileNetV2". The context and composite
diagrams name the component "OpenCV / ML Kit". These describe three different technologies with
incompatible deployment models.

**Decision.** Server-side single-dish image classification with **EfficientNet-B0** (primary), with
**MobileNetV2** trained and reported as the baseline. The model runs in the FastAPI backend through ONNX
Runtime. OpenCV/Pillow is used only for image decoding and preprocessing. ML Kit and YOLO are removed.

**Replacement text.**

* Context diagram and composite diagram: rename "OpenCV / ML Kit" to **"Food Recognition Service
  (FastAPI; EfficientNet-B0 classifier via ONNX Runtime; OpenCV/Pillow preprocessing)"**.
* Sequence diagram 5.4.3: rename "YOLOv4 / MobileNetV2" to **"Food Recognition Service
  (EfficientNet-B0)"**. Change the returned message from "label, bounding box, confidence" to
  **"top-3 {label, confidence}, band, bbox = null"**.
* 3.1.5 / FR-SCAN: wherever the system "detects items with bounding boxes", write: **"The system shall
  classify the dominant dish in the photo and return the three most likely dishes with calibrated
  confidences. Bounding boxes are optional and are not returned in this version."** Add: **"The user
  can add further items from the same meal through manual food search."**

**Justification (for the report).**

1. *ML Kit runs on the device.* It is Google's on-device mobile SDK. The spec places inference on the
   server and lists on-device/offline AI as out of scope, so ML Kit contradicts the architecture. Its
   generic image labeller also does not know dishes such as haleem or chanay.
2. *OpenCV is not a recognition model.* It is an image-processing library. Naming it as the recognition
   component overstates what it does. It stays in the design only for preprocessing.
3. *YOLO is a detector and needs box annotations.* Detection needs every training image annotated with
   bounding boxes per item. Public South Asian food datasets are image-level labelled only, and drawing
   boxes on thousands of images was not feasible in the project time. Detection also targets multi-item
   plates, while the agreed scope is one dominant dish per photo plus manual "add item". In addition,
   the maintained YOLO implementations (Ultralytics YOLOv5/v8/11) are AGPL-3.0.
4. *Why EfficientNet-B0.* It is accurate for its size, transfer-learns well from ImageNet, runs on a CPU
   server within the latency budget (see results), and exports cleanly to ONNX. Keeping MobileNetV2 as
   a trained baseline preserves the spec's original model name as a measured comparison rather than a
   claim.
5. *One name everywhere* removes ambiguity for implementers and examiners.

## 2. Human-in-the-loop verification (Phase 2 contract, agreed)

* `POST /detect-food` **never writes** a meal log. It returns suggestions, a band and a `detectionId`.
* Only `POST /meal-logs` writes, and only with explicitly confirmed items **and a meal type**.
* The server recomputes every calorie value from `foodId` and grams. Client-sent totals are ignored.
* Firestore security rules deny client writes to meal logs. Only the backend (Admin SDK) writes.
* `detectionId` links a log to the scan that produced it, for audit and for measuring
  accept/edit/reject rates.
* Honest limit: the server cannot prove that a human looked at the screen. It guarantees that nothing is
  logged without an explicit confirm request carrying a meal type.

## 3. Meal type: impact of the new requirement

The user must choose BREAKFAST, LUNCH, DINNER or **SNACK** (SNACK agreed) after verification. The app
pre-selects one from the local time (PKT, UTC+5): 04:00-10:59 breakfast, 11:00-15:59 lunch,
16:00-03:59 dinner. The user can always change it, and nothing can be logged without it.

| Area | Change |
|---|---|
| API schema | `POST /meal-logs` body gains required `mealType` (enum) and `mealDate` (user's local date). Responses and `GET /meal-logs?date=` return it. |
| Validation | Missing or unknown `mealType` -> 422. No server-side default (the time-of-day default is a UI suggestion only). `mealDate` must be within a sane window of the server time once the client's timezone offset is applied. |
| Data model | `MealLog.mealType`, `mealDate` (local), `loggedAt` (UTC), `tzOffsetMinutes`. Each item stores `foodId`, grams and server-computed nutrients. |
| Adaptive re-planner | Adherence is measured per (date, mealType). Logged kcal and macros are compared with the planned breakfast, lunch and dinner. SNACK counts toward daily totals and is matched only if the plan contains snacks; it is never forced into lunch or dinner. |
| Class diagram | Add `MealLog`, `MealLogItem`, `Detection` and enum `MealType` (below). |
| ERD / Firestore | Add `users/{uid}/mealLogs/{logId}` and `detections/{detectionId}` (below). |
| App UI | Meal-type selector on the confirm screen, pre-selected by time, mandatory. Track and Progress views group logs by meal type. |
| Security rules | `mealLogs`: owner read, no client write. `detections`: no client access. |

### Class diagram additions

```
enum MealType { BREAKFAST, LUNCH, DINNER, SNACK }

class MealLog {
  logId: String
  userId: String
  mealType: MealType          <<required>>
  mealDate: Date              (user's local date)
  loggedAt: DateTime          (UTC)
  tzOffsetMinutes: int
  detectionId: String [0..1]
  items: MealLogItem [1..*]
  totalKcal, totalProteinG, totalCarbsG, totalFatG: double   (server-computed)
}
class MealLogItem {
  foodId: String              (must exist in foods)
  portionGrams: double
  kcal, proteinG, carbsG, fatG: double   (server-computed)
  origin: {scan_confirmed, scan_edited, manual}
}
class Detection {
  detectionId: String
  userId: String
  createdAt, expiresAt: DateTime
  modelVersion: String
  band: {high, medium, low}
  suggestions: {label, foodId, confidence} [0..3]
}
User "1" -- "*" MealLog
MealLog "1" *-- "1..*" MealLogItem
MealLogItem "*" --> "1" Food
MealLog "*" --> "0..1" Detection
DietPlan ..> MealType : planned meals keyed by
```

### ERD (Firestore) additions

```
users/{uid}/mealLogs/{logId}
    mealType   string  enum BREAKFAST|LUNCH|DINNER|SNACK   (required)
    mealDate   string  YYYY-MM-DD (local)
    loggedAt   timestamp (UTC)
    tzOffsetMinutes number
    detectionId string | null    -> detections/{detectionId}
    items      array<{foodId -> foods/{foodId}, portionGrams, kcal, proteinG, carbsG, fatG, origin}>
    totals     map {kcal, proteinG, carbsG, fatG}
detections/{detectionId}
    userId, createdAt, expiresAt (24 h), modelVersion, band, suggestions[]   (no image stored)
foods/{foodId}   (existing, admin-managed; gains optional cvLabel or is mapped by a backend JSON file)
```
