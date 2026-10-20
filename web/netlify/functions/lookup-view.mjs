import { getStore } from "@netlify/blobs";
import { lookupView } from "../lib/shared-views.mjs";

export default (request) => lookupView(request, getStore({ name: "shared-views-v1", consistency: "strong" }));

export const config = { path: "/api/views/:digest" };
