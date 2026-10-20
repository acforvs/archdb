import { getStore } from "@netlify/blobs";
import { openView } from "../lib/shared-views.mjs";

export default (request) => openView(request, getStore({ name: "shared-views-v1", consistency: "strong" }));

export const config = {
  path: ["/:code", "/:code/"], preferStatic: true,
  rateLimit: { windowLimit: 120, windowSize: 60, aggregateBy: ["ip", "domain"] },
};
