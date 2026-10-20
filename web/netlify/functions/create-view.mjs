import { getStore } from "@netlify/blobs";
import { createView } from "../lib/shared-views.mjs";

export default (request) => createView(request, getStore({ name: "shared-views-v1", consistency: "strong" }));

export const config = {
  path: "/api/views",
  rateLimit: { windowLimit: 30, windowSize: 60, aggregateBy: ["ip", "domain"] },
};
