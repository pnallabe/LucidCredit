/**
 * frontend/src/lib/utils.ts
 * Shared utility: class name merger (tailwind-merge + clsx).
 */
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
