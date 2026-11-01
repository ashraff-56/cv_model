# How photo calorie apps estimate food quantity (research notes, 1 Nov 2026)

**Question.** SmartBite's users (gym-goers tracking protein, carbs and fat) will not enter quantities. How do
existing apps get from one photo to grams and nutrients, and how accurate is that?

Sources are split into primary (papers, the company's own statements, official docs) and secondary
(third-party reviews). Secondary claims are marked *unverified*.

## 1. What the apps do

| App | Recognition | Quantity | User step | Evidence |
|---|---|---|---|---|
| **Cal AI** | Cloud vision LLMs: "models from Anthropic and OpenAI and RAG"; "different models are better with different foods" | Estimated by the model from the photo (a third-party explainer also claims depth-sensor use, *unverified* and not in the founders' statements) | Results editable | TechCrunch interview with the founders [1]; claimed "90% accurate" (company claim, no published test) |
| **MyFitnessPal Meal Scan** | Passio Nutrition-AI SDK (camera stream → recognised foods + nutrition) | Passio advertises "food amount estimation technology"; a third-party benchmark says it uses a lookup table of average portion weights (*unverified*) | User confirms foods, portion and nutrition | Passio docs and SDK [2], Passio case study [3] |
| **SnapCalorie** | Vision model (team from Google Lens / Cloud Vision; co-founder co-authored Nutrition5k) | **Depth (LiDAR) volume** on supported iPhones | Editable | App Store listing [4], Nutrition5k [5] |
| **LogMeal / Foodvisor** (APIs) | Segmentation of each item on the plate, then per-item recognition | "Estimate portion sizes from images; return volume/serving per item"; RGB, or RGB + depth for better precision | Quantity-confirmation endpoint exists | LogMeal API docs [6] |

**Common pattern.** Every product splits the task in three: **what** is on the plate (recognition,
usually per item), **how much** (a portion estimate), and **nutrients per gram** (a food-composition
database such as USDA FDC). Every product also lets the user confirm or edit, because the quantity step is
the weak link.

## 2. What the measurements say

**Nutrition5k** (Google, CVPR 2021) [5]: 5,006 cafeteria plates with every ingredient weighed; Creative Commons 4.0.
Test-set mean absolute error (Table 3), all with a fixed camera rig 35.9 cm above the plate:

| Method | Calories | Total mass | Protein | Fat | Carbs |
|---|---|---|---|---|---|
| Always predict the mean | 60.2% | 58.5% | 62.1% | 67.6% | 62.1% |
| **RGB photo only** | **26.1%** | **18.8%** | **29.5%** | 34.2% | 31.9% |
| RGB + depth (4th channel) | 18.8% | 18.9% | 20.9% | 18.1% | 23.8% |
| Depth-derived volume + per-gram model | 16.5% | 13.7% | 31.1% | 25.5% | 22.0% |
| Per-gram only (food known, weight given) | 9.5% per g | n/a | 12.0% per g | 18.3% per g | 13.9% per g |

Knowing *what* the food is gives ~10% error per gram; having to guess *how much* from a photo roughly
triples it. That is in a controlled rig; hand-held phone photos (unknown distance and angle) are harder.

**Independent tests of real apps.**

* NIH Clinical Center, presented at NUTRITION 2026 (conference result, not yet a journal paper) [7]:
  MyFitnessPal, Lose It!, Cal AI and Appediet on 102 weighed meals underestimated calories by about
  **250–345 kcal per meal** and fat by about **30 g**. Carbohydrates were estimated more consistently than fat
  or protein. Oil, butter and dressings are invisible in a photo.
* ChatGPT-5 on 195 dishes (*Nutrients* 2025) [8]: energy error **30.5% from the photo alone**, 24.4% with
  basic descriptors, 13.9% with the photo plus an ingredient list.

## 3. What this means for SmartBite

1. **Recognition is the trainable, measurable part.** SmartBite's classifier plus the per-100 g nutrition table
   is the "per-gram" half of the Nutrition5k decomposition.
2. **Quantity from a single phone photo is the hard part.** The best published RGB-only result is ~19% mass
   error in a fixed rig, and real apps miss about a third of calories in independent tests.
3. **Pakistani food is a worst case for the hidden-fat problem.** Ghee and oil in curries, parathas and
   halwa don't show in photos. Recipe-based per-100 g values (which include typical oil) help; a photo
   alone cannot tell a light from a heavy hand with ghee.
4. **Multi-item plates** (rice + chicken + salad) need per-item detection or segmentation. There is no
   public Pakistani dataset with boxes or masks, so that needs either annotation work or a general-purpose
   vision model.
5. **Every app keeps a confirm/adjust step.** The NIH result is about users who did *not* adjust portions.

## 4. Options for the quantity step

| Option | How | Can we measure it? | Main risk |
|---|---|---|---|
| A. Typical serving (auto) | Dish → standard serving weight (FNDDS/INDB portions, Pakistani serving sizes) → nutrients; user may tap S/M/L | Only the serving table; no image-based measurement | Ignores the actual amount on the plate |
| B. Trained portion model | Image → grams regressor on Nutrition5k (+ per-gram nutrients from our classifier/DB) | Yes, on Nutrition5k's test set | US cafeteria food and a fixed camera height; phone photos of Pakistani food will be worse and can't be checked without weighed meals |
| C. Vision-LLM estimator (Cal AI style) | Photo → Claude lists items + grams; server computes nutrients from our DB (not the LLM's numbers) | Yes, on Nutrition5k test photos (grams) | Per-photo API cost, internet, privacy; still ~30% calorie error in published LLM tests |

## References

[1] TechCrunch, "Photo calorie app Cal AI, downloaded over a million times, was built by two
teenagers," Mar. 16, 2025. https://techcrunch.com/2025/03/16/photo-calorie-app-cal-ai-downloaded-over-a-million-times-was-built-by-two-teenagers/

[2] Passio, "Nutrition-AI documentation," and "Passio Nutrition AI Android SDK Distribution" (GitHub).
https://passio.gitbook.io/nutrition-ai ; https://github.com/Passiolife/Passio-Nutrition-AI-Android-SDK-Distribution

[3] Passio, "MyFitnessPal — Nutrition AI by Passio AI" (case study). https://www.passio.ai/case-studies/myfitnesspal

[4] SnapCalorie, "SnapCalorie AI Calorie Counter," Apple App Store. https://apps.apple.com/us/app/snapcalorie-ai-calorie-counter/id1574239307

[5] Q. Thames, A. Karpur, W. Norris, F. Xia, L. Panait, T. Weyand, and J. Sim, "Nutrition5k: Towards automatic
nutritional understanding of generic food," in *Proc. IEEE/CVF CVPR*, 2021, pp. 8903–8911. Dataset:
https://github.com/google-research-datasets/Nutrition5k ("We release all Nutrition5k data under the Creative Commons V4.0 license"; the variant is not named).

[6] LogMeal, "Features overview" and "Food quantity confirmation," LogMeal API documentation.
https://docs.logmeal.com/docs/guides-features-overview

[7] ScienceDaily (American Society for Nutrition release), "Your AI calorie-tracking app may be off by 345
calories," Jul. 2026; results presented at NUTRITION 2026. https://www.sciencedaily.com/releases/2026/07/260726015237.htm

[8] "Image-Based Dietary Energy and Macronutrients Estimation with ChatGPT-5: Cross-Source Evaluation Across
Escalating Context Scenarios," *Nutrients*, vol. 17, no. 22, p. 3613, 2025. doi:10.3390/nu17223613

Not checked: app internals (no APK analysis was done); author lists for [1] and [8].
